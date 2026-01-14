import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import ray
from ray.experimental.gpu_object_manager.tensor_transport_manager import (
    TensorTransportManager,
)
from ray.experimental.gpu_object_manager.types import (
    CommunicatorMetadata,
    TensorTransportMetadata,
)
import logging

if TYPE_CHECKING:
    import torch


@dataclass
class UCCLCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the UCCL communicator.

    Args:
        receiver_oob_ip: The OOB IP address of the receiver.
        receiver_listen_port: The TCP listen port for OOB coordination on the receiver.
    """
    receiver_oob_ip: Optional[str] = None
    receiver_listen_port: Optional[int] = None


@dataclass
class UCCLTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for UCCL transport.

    Args:
        sender_oob_ip: The OOB IP address of the sender.
        sender_listen_port: The TCP listen port for OOB coordination on the sender.
    """
    sender_oob_ip: Optional[str] = None
    sender_listen_port: Optional[int] = None

    __eq__ = object.__eq__
    __hash__ = object.__hash__


class UCCLTensorTransport(TensorTransportManager):
    def __init__(self, tensor_transport_backend: str):
        # Lazy initialization to avoid importing UCCL unnecessarily
        self._transfer_manager = None
        self._aborted_transfer_obj_ids = set()
        self._aborted_transfer_obj_ids_lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    @property
    def tensor_transport_backend(self) -> str:
        return "UCCL"

    @staticmethod
    def is_one_sided() -> bool:
        # Using one-sided RDMA WRITE via TransferManager
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return True

    def _get_transfer_manager(self):
        """
        Creates a UCCL TransferManager if not already created.
        """
        if self._transfer_manager is not None:
            return self._transfer_manager

        from uccl.transfer import TransferManager

        # Get GPU index from Ray runtime context
        ray_gpu_ids = ray.get_gpu_ids()
        if ray_gpu_ids:
            # Important: Use local CUDA device index (0), not Ray's physical GPU ID
            # Ray sets CUDA_VISIBLE_DEVICES, so the assigned GPU is always device 0 locally
            local_gpu_idx = 0
        else:
            local_gpu_idx = 0
            self.logger.warning("[UCCL] No GPUs assigned via Ray, defaulting to GPU index 0")

        # Use a reasonable default for num_cpus
        num_cpus = 4

        # Use port 0 to let OS assign a free port
        listen_port = 0

        self._transfer_manager = TransferManager(local_gpu_idx, num_cpus, listen_port)

        # Get the actual bound port (since we passed 0)
        actual_port = self._transfer_manager.listen_socket.getsockname()[1]
        self.logger.info(f"[UCCL] TransferManager initialized on GPU {local_gpu_idx}, listening on port {actual_port}")

        return self._transfer_manager

    def _get_actual_listen_port(self) -> int:
        """Get the actual TCP listen port from the TransferManager."""
        transfer_manager = self._get_transfer_manager()
        return transfer_manager.listen_socket.getsockname()[1]

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        """Check if the remote actor has UCCL tensor transport available."""
        def __ray_actor_has_tensor_transport__(_self) -> bool:
            try:
                from ray.experimental.gpu_object_manager.util import (
                    get_tensor_transport_manager,
                )
                transport_manager = get_tensor_transport_manager("UCCL")
                transport_manager._get_transfer_manager()
                return True
            except Exception as e:
                print(f"[UCCL DEBUG] Exception checking tensor transport: {e}", flush=True)
                return False

        return ray.get(
            actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __ray_actor_has_tensor_transport__
            )
        )

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["torch.Tensor"],
    ) -> UCCLTransportMetadata:
        """Extract metadata needed for UCCL transport from GPU tensors.

        For TransferManager-based implementation, we store the sender's OOB IP
        and listen port. Memory registration happens dynamically during transfer.
        """
        from ray._private.worker import global_worker

        gpu_object_store = global_worker.gpu_object_manager.gpu_object_store
        device = None
        tensor_meta = []

        # Check for duplicate metadata
        duplicate_meta = gpu_object_store.record_and_get_meta_if_duplicate(
            obj_id, gpu_object
        )
        if duplicate_meta is not None:
            return duplicate_meta

        if gpu_object:
            from uccl import p2p

            # Initialize the TransferManager to get the listen port
            transfer_manager = self._get_transfer_manager()

            # Get the OOB IP and actual listen port for coordination
            local_metadata = transfer_manager.ep.get_metadata()
            sender_oob_ip = p2p.Endpoint.parse_metadata(local_metadata)[0]
            sender_listen_port = self._get_actual_listen_port()

            # We assume all tensors in one GPU object have the same device type
            device = gpu_object[0].device
            for t in gpu_object:
                if t.device.type != device.type:
                    raise ValueError(
                        "All tensors in an RDT object must have the same device type."
                    )
                tensor_meta.append((t.shape, t.dtype))
        else:
            sender_oob_ip, sender_listen_port = None, None

        ret = UCCLTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=device,
            sender_oob_ip=sender_oob_ip,
            sender_listen_port=sender_listen_port,
        )
        # Cache the metadata using the generic API
        gpu_object_store.record_managed_meta("UCCL", obj_id, ret)
        return ret

    def get_communicator_metadata(
        self,
        _src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        _backend: Optional[str] = None,
    ) -> UCCLCommunicatorMetadata:
        """Get communicator metadata for UCCL transport.

        For TransferManager-based implementation, we need the receiver's
        OOB IP and listen port so the sender can connect to the receiver.
        """
        def __get_receiver_oob_info__(_self):
            """Helper function to get receiver's OOB IP and listen port."""
            from ray.experimental.gpu_object_manager.util import (
                get_tensor_transport_manager,
            )
            from uccl import p2p

            transport_manager = get_tensor_transport_manager("UCCL")
            transfer_manager = transport_manager._get_transfer_manager()  # Initialize TransferManager

            local_metadata = transfer_manager.ep.get_metadata()
            oob_ip = p2p.Endpoint.parse_metadata(local_metadata)[0]
            listen_port = transport_manager._get_actual_listen_port()

            return oob_ip, listen_port

        # Call on dst_actor (receiver) to get its OOB coordination info
        receiver_oob_ip, receiver_listen_port = ray.get(
            dst_actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __get_receiver_oob_info__
            )
        )

        print("[Warning], this is unlikely to be called.")
        return UCCLCommunicatorMetadata(
            receiver_oob_ip=receiver_oob_ip,
            receiver_listen_port=receiver_listen_port
        )

    def recv_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        obj_id: str,
        tensor_transport_metadata: UCCLTransportMetadata,
        communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Receive multiple tensors using UCCL one-sided transport.

        The receiver acts as the TARGET/SERVER:
        1. Accept connection from sender
        2. Register tensors and post metadata (advertise memory)
        3. Wait for sender to complete the transfer
        """
        if not tensors:
            return

        assert isinstance(
            tensor_transport_metadata, UCCLTransportMetadata
        ), "metadata must be a UCCLTransportMetadata object for UCCL transport"
        assert isinstance(
            communicator_metadata, UCCLCommunicatorMetadata
        ), "metadata must be a UCCLCommunicatorMetadata object for UCCL transport"

        # Check if transfer was aborted before starting
        with self._aborted_transfer_obj_ids_lock:
            if obj_id in self._aborted_transfer_obj_ids:
                self._aborted_transfer_obj_ids.remove(obj_id)
                raise RuntimeError(f"UCCL transfer aborted for object id: {obj_id}")

        transfer_manager = self._get_transfer_manager()
        transfer_ids = []
        conn_id = None

        try:
            # Step 1: Accept connection from sender (blocking)
            self.logger.info(f"[UCCL Receiver] Waiting for connection from sender...")
            conn_id = transfer_manager.accept()
            self.logger.info(f"[UCCL Receiver] Connection established, conn_id={conn_id}")

            # Step 2: For each tensor, register and post metadata
            for i, tensor in enumerate(tensors):
                # Check for abort
                with self._aborted_transfer_obj_ids_lock:
                    if obj_id in self._aborted_transfer_obj_ids:
                        self._aborted_transfer_obj_ids.remove(obj_id)
                        raise RuntimeError(f"UCCL transfer aborted for object id: {obj_id}")

                if not tensor.is_contiguous():
                    raise ValueError(f"Tensor {i} must be contiguous for UCCL transfer")

                # Register the tensor with TransferManager
                transfer_id = transfer_manager.register_transfer(conn_id, tensor)
                transfer_ids.append(transfer_id)
                self.logger.info(f"[UCCL Receiver] Registered tensor {i}, transfer_id={transfer_id}")

                # Post transfer metadata (advertise memory -> sends fifo_blob to sender over TCP)
                transfer_manager.post_transfer_metadata(transfer_id)
                self.logger.info(f"[UCCL Receiver] Posted metadata for tensor {i}")

                # Wait for sender to complete this transfer and signal done
                self.logger.info(f"[UCCL Receiver] Waiting for transfer {i} to complete...")
                transfer_manager.wait_transfer_done(transfer_id)
                self.logger.info(f"[UCCL Receiver] Transfer {i} completed")

        finally:
            # Best effort cleanup
            with self._aborted_transfer_obj_ids_lock:
                self._aborted_transfer_obj_ids.discard(obj_id)

            # Deregister all transfers
            for transfer_id in transfer_ids:
                try:
                    transfer_manager.deregister_transfer(transfer_id)
                except Exception as e:
                    self.logger.warning(f"[UCCL Receiver] Failed to deregister transfer {transfer_id}: {e}")

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: UCCLTransportMetadata,
        communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Send multiple tensors using UCCL one-sided transport.

        The sender acts as the INITIATOR/CLIENT:
        1. Connect to receiver
        2. Register tensors
        3. Fetch metadata from receiver and perform one-sided WRITE
        4. Signal completion to receiver
        """
        if not tensors:
            return

        assert isinstance(
            tensor_transport_metadata, UCCLTransportMetadata
        ), "metadata must be a UCCLTransportMetadata object for UCCL transport"
        assert isinstance(
            communicator_metadata, UCCLCommunicatorMetadata
        ), "metadata must be a UCCLCommunicatorMetadata object for UCCL transport"

        # Get receiver's OOB info from communicator metadata
        receiver_oob_ip = communicator_metadata.receiver_oob_ip
        receiver_listen_port = communicator_metadata.receiver_listen_port

        if not receiver_oob_ip or not receiver_listen_port:
            raise RuntimeError("[UCCL Sender] Receiver OOB info not available in communicator metadata")

        transfer_manager = self._get_transfer_manager()
        transfer_ids = []
        conn_id = None

        try:
            # Step 1: Connect to receiver
            self.logger.info(f"[UCCL Sender] Connecting to receiver at {receiver_oob_ip}:{receiver_listen_port}...")
            conn_id = transfer_manager.connect(receiver_oob_ip, receiver_listen_port)
            self.logger.info(f"[UCCL Sender] Connection established, conn_id={conn_id}")

            # Step 2: For each tensor, register, fetch metadata, transfer, and signal done
            for i, tensor in enumerate(tensors):
                if not tensor.is_contiguous():
                    raise ValueError(f"Tensor {i} must be contiguous for UCCL transfer")

                # Register the tensor with TransferManager
                transfer_id = transfer_manager.register_transfer(conn_id, tensor)
                transfer_ids.append(transfer_id)
                self.logger.info(f"[UCCL Sender] Registered tensor {i}, transfer_id={transfer_id}")

                # Fetch transfer metadata from receiver (receives fifo_blob over TCP)
                transfer_metadata = transfer_manager.fetch_transfer_metadata(transfer_id)
                self.logger.info(f"[UCCL Sender] Fetched metadata for tensor {i}")

                # Perform one-sided WRITE transfer
                poll_id = transfer_manager.do_transfer_async(transfer_id, transfer_metadata)
                self.logger.info(f"[UCCL Sender] Started transfer for tensor {i}, poll_id={poll_id}")

                # Poll until transfer completes
                while not transfer_manager.check_transfer_done(transfer_id, poll_id):
                    time.sleep(0.001)

                self.logger.info(f"[UCCL Sender] Transfer {i} completed")

                # Signal completion to receiver
                transfer_manager.post_transfer_done(transfer_id)
                self.logger.info(f"[UCCL Sender] Posted done signal for tensor {i}")

        except Exception as e:
            self.logger.error(f"[UCCL Sender] Send failed: {e}")
            raise

        finally:
            # Deregister all transfers
            for transfer_id in transfer_ids:
                try:
                    transfer_manager.deregister_transfer(transfer_id)
                except Exception as e:
                    self.logger.warning(f"[UCCL Sender] Failed to deregister transfer {transfer_id}: {e}")

    def garbage_collect(
        self, obj_id: str, tensor_transport_meta: UCCLTransportMetadata
    ):
        """Clean up resources associated with a tensor transport.

        With TransferManager, memory registration/deregistration is handled
        automatically during register_transfer/deregister_transfer calls,
        so we only need to clean up the metadata cache.
        """
        from ray._private.worker import global_worker

        gpu_object_store = global_worker.gpu_object_manager.gpu_object_store
        gpu_object_store.remove_managed_meta("UCCL", obj_id)

    def abort_transport(
        self,
        obj_id: str,
        _communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Abort an ongoing transfer for the given object ID."""
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.add(obj_id)
