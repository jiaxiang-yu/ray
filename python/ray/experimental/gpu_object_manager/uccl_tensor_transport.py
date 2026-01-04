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
        receiver_endpoint_metadata: Serialized endpoint metadata of the receiver (IP, port, logical GPU index).
        receiver_ray_physical_gpu_id: The physical GPU ID of the receiver as assigned by Ray (for IPC determination).
    """
    receiver_endpoint_metadata: Optional[bytes] = None
    receiver_ray_physical_gpu_id: Optional[int] = None


@dataclass
class UCCLTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for UCCL transport.

    Args:
        uccl_endpoint_metadata: Serialized endpoint metadata (IP, port, logical GPU index).
        uccl_mr_ids: Memory region IDs for registered tensors.
        tensor_ptrs: Data pointers for the tensors.
        ray_physical_gpu_id: The physical GPU ID as assigned by Ray (for IPC determination).
    """
    uccl_endpoint_metadata: Optional[bytes] = None
    uccl_mr_ids: Optional[List[int]] = None
    tensor_ptrs: Optional[List[int]] = None
    ray_physical_gpu_id: Optional[int] = None

    __eq__ = object.__eq__
    __hash__ = object.__hash__

class UCCLTensorTransport(TensorTransportManager):
    def __init__(self, tensor_transport_backend: str):
        # Lazy initialization to avoid importing UCCL unnecessarily
        self._uccl_endpoint = None
        self._aborted_transfer_obj_ids = set()
        self._aborted_transfer_obj_ids_lock = threading.Lock()

        self.logger = logging.getLogger(__name__)

    @property
    def tensor_transport_backend(self) -> str:
        return "UCCL"

    @staticmethod
    def is_one_sided() -> bool:
        # send/recv in UCCL are two-sided
        return False

    @staticmethod
    def can_abort_transport() -> bool:
        return True

    def _get_uccl_endpoint(self):
        """
        Creates a UCCL endpoint if not already created.
        """
        if self._uccl_endpoint is not None:
            return self._uccl_endpoint

        from uccl.p2p import Endpoint

        # endpoint require a local gpu index and a number of CPUs
        # Get GPU index from Ray runtime context
        ray_gpu_ids = ray.get_gpu_ids()
        if ray_gpu_ids:
            # Important: Use local CUDA device index (0), not Ray's physical GPU ID
            # Ray sets CUDA_VISIBLE_DEVICES, so the assigned GPU is always device 0 locally
            # TODO: verify if this is the correct way.
            local_gpu_idx = 0
        else:
            local_gpu_idx = 0
            self.logger.warning("[UCCL] No GPUs assigned via Ray, defaulting to GPU index 0")
        # Use a reasonable default for num_cpus
        num_cpus = 4
        self._uccl_endpoint = Endpoint(local_gpu_idx, num_cpus)
        return self._uccl_endpoint

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        """Check if the remote actor has UCCL tensor transport available."""
        def __ray_actor_has_tensor_transport__(_self) -> bool:
            try:
                from ray.experimental.gpu_object_manager.util import (
                    get_tensor_transport_manager,
                )
                get_tensor_transport_manager("UCCL")._get_uccl_endpoint()
                return True
            except Exception:
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
        """Extract metadata needed for UCCL transport from GPU tensors."""
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

        # Get the physical GPU ID from Ray for IPC determination
        ray_gpu_ids = ray.get_gpu_ids()
        ray_physical_gpu_id = ray_gpu_ids[0] if ray_gpu_ids else None

        if gpu_object:
            endpoint = self._get_uccl_endpoint()
            endpoint_metadata = endpoint.get_metadata()

            # Register memory for each tensor
            mr_ids = []
            tensor_ptrs = []

            # We assume all tensors in one GPU object have the same device type
            device = gpu_object[0].device
            for t in gpu_object:
                if t.device.type != device.type:
                    raise ValueError(
                        "All tensors in an RDT object must have the same device type."
                    )

                # Register the tensor memory with UCCL
                ptr = t.data_ptr()
                size = t.numel() * t.element_size()
                ok, mr_id = endpoint.reg(ptr, size)
                if not ok:
                    raise RuntimeError(f"Failed to register memory for tensor in object {obj_id}")

                mr_ids.append(mr_id)
                tensor_ptrs.append(ptr)
                tensor_meta.append((t.shape, t.dtype))
        else:
            endpoint_metadata, mr_ids, tensor_ptrs = None, None, None

        ret = UCCLTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=device,
            uccl_endpoint_metadata=endpoint_metadata,
            uccl_mr_ids=mr_ids,
            tensor_ptrs=tensor_ptrs,
            ray_physical_gpu_id=ray_physical_gpu_id,
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

        For one-sided transport, we need the receiver's endpoint metadata
        so the sender can determine IPC vs RDMA before accepting the connection.
        """
        def __get_receiver_endpoint_metadata__(_self):
            """Helper function to get receiver's endpoint metadata."""
            from ray.experimental.gpu_object_manager.util import (
                get_tensor_transport_manager,
            )
            transport_manager = get_tensor_transport_manager("UCCL")
            endpoint = transport_manager._get_uccl_endpoint()
            endpoint_metadata = endpoint.get_metadata()
            ray_gpu_ids = ray.get_gpu_ids()
            ray_physical_gpu_id = ray_gpu_ids[0] if ray_gpu_ids else None
            return endpoint_metadata, ray_physical_gpu_id

        # Call on dst_actor (receiver) to get its endpoint metadata
        receiver_meta, receiver_gpu_id = ray.get(
            dst_actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __get_receiver_endpoint_metadata__
            )
        )

        return UCCLCommunicatorMetadata(
            receiver_endpoint_metadata=receiver_meta,
            receiver_ray_physical_gpu_id=receiver_gpu_id
        )

    def recv_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        obj_id: str,
        tensor_transport_metadata: UCCLTransportMetadata,
        communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Receive multiple tensors using UCCL one-sided transport."""
        if not tensors:
            return

        assert isinstance(
            tensor_transport_metadata, UCCLTransportMetadata
        ), "metadata must be a UCCLTransportMetadata object for UCCL transport"
        assert isinstance(
            communicator_metadata, UCCLCommunicatorMetadata
        ), "metadata must be a UCCLCommunicatorMetadata object for UCCL transport"

        sender_endpoint_metadata = tensor_transport_metadata.uccl_endpoint_metadata
        sender_ray_physical_gpu_id = tensor_transport_metadata.ray_physical_gpu_id

        # Check if transfer was aborted before starting
        with self._aborted_transfer_obj_ids_lock:
            if obj_id in self._aborted_transfer_obj_ids:
                self._aborted_transfer_obj_ids.remove(obj_id)
                raise RuntimeError(f"UCCL transfer aborted for object id: {obj_id}")

        endpoint = self._get_uccl_endpoint()
        local_mr_ids = []
        transfer_handles = []

        try:
            # Parse sender's endpoint info
            from uccl.p2p import Endpoint
            import socket
            self.logger.info(f"[UCCL Receiver] DEBUG ON!")

            sender_ip, sender_port, sender_gpu = Endpoint.parse_metadata(sender_endpoint_metadata)

            # Get local info
            local_metadata = endpoint.get_metadata()

            local_ip, _local_port, local_gpu = Endpoint.parse_metadata(local_metadata)
            local_hostname = socket.gethostname()
            sender_hostname = socket.getfqdn(sender_ip)

            # Determine if IPC (same machine)
            # Use IPC for all same-machine connections to avoid RDMA NIC routing issues
            same_machine = (sender_ip == local_ip or sender_ip == "127.0.0.1" or local_hostname == sender_hostname)
            is_ipc = same_machine

            self.logger.info(f"[UCCL Receiver] Using {'IPC' if is_ipc else 'RDMA'} for receiving (sender_ip={sender_ip}, local_ip={local_ip})")

            if is_ipc:
                # IPC: Accept connection from local sender (receiver waits for sender)
                self.logger.info(f"[UCCL Receiver] Waiting for IPC connection from sender on GPU {local_gpu}...")
                ok, remote_gpu_idx, conn_id = endpoint.accept_local()
                self.logger.info(f"[UCCL Receiver] Accepted IPC connection: ok={ok}, remote_gpu={remote_gpu_idx}, conn_id={conn_id}")
                if not ok:
                    raise RuntimeError(f"[UCCL Receiver] Failed to accept IPC connection from sender")

                # Receive tensors via IPC (no memory registration needed)
                for i, tensor in enumerate(tensors):
                    ptr = tensor.data_ptr()
                    size = tensor.numel() * tensor.element_size()
                    self.logger.info(f"[UCCL Receiver] Receiving tensor {i} via IPC: size={size} bytes")
                    ok, transfer_id = endpoint.recv_ipc_async(conn_id, ptr, size)
                    if not ok:
                        raise RuntimeError(f"[UCCL Receiver] Failed to initiate IPC recv for tensor {i}")
                    transfer_handles.append(transfer_id)
            else:
                # RDMA: Accept connection from remote sender (receiver waits for sender)
                self.logger.info(f"[UCCL Receiver] Waiting for RDMA connection from sender at {sender_ip}:{sender_port}...")
                ok, remote_ip, remote_gpu, conn_id = endpoint.accept()
                self.logger.info(f"[UCCL Receiver] RDMA connection accepted from {remote_ip}, GPU {remote_gpu}, conn_id={conn_id}")
                if not ok:
                    raise RuntimeError(f"[UCCL Receiver] Failed to accept RDMA connection from sender")

                # Register local memory for RDMA receive
                for tensor in tensors:
                    ptr = tensor.data_ptr()
                    size = tensor.numel() * tensor.element_size()
                    ok, mr_id = endpoint.reg(ptr, size)
                    if not ok:
                        raise RuntimeError("[UCCL Receiver] Failed to register local memory for RDMA receiving")
                    local_mr_ids.append((mr_id, ptr, size))

                # Receive tensors via RDMA
                for i, (local_mr_id, local_ptr, size) in enumerate(local_mr_ids):
                    self.logger.info(f"[UCCL Receiver] Receiving tensor {i} via RDMA: mr_id={local_mr_id}, size={size} bytes")
                    ok, transfer_id = endpoint.recv_async(conn_id, local_mr_id, local_ptr, size)
                    if not ok:
                        raise RuntimeError(f"[UCCL Receiver] Failed to initiate RDMA recv for tensor {i}")
                    transfer_handles.append(transfer_id)

            # Poll all transfers until completion
            for i, transfer_id in enumerate(transfer_handles):
                while True:
                    # Check for abort
                    with self._aborted_transfer_obj_ids_lock:
                        if obj_id in self._aborted_transfer_obj_ids:
                            self._aborted_transfer_obj_ids.remove(obj_id)
                            raise RuntimeError(f"UCCL transfer aborted for object id: {obj_id}")

                    ok, is_done = endpoint.poll_async(transfer_id)
                    if not ok:
                        raise RuntimeError(f"[UCCL Receiver] Error polling transfer {i} (id={transfer_id})")

                    if is_done:
                        break

                    time.sleep(0.001)

        finally:
            # Best effort cleanup
            with self._aborted_transfer_obj_ids_lock:
                self._aborted_transfer_obj_ids.discard(obj_id)

            # Deregister local memory regions (RDMA only)
            for mr_id, _, _ in local_mr_ids:
                endpoint.dereg(mr_id)

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: UCCLTransportMetadata,
        communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Send multiple tensors using UCCL transport."""
        if not tensors:
            return

        assert isinstance(
            tensor_transport_metadata, UCCLTransportMetadata
        ), "metadata must be a UCCLTransportMetadata object for UCCL transport"
        assert isinstance(
            communicator_metadata, UCCLCommunicatorMetadata
        ), "metadata must be a UCCLCommunicatorMetadata object for UCCL transport"

        # The sender already has the tensors registered during metadata extraction
        mr_ids = tensor_transport_metadata.uccl_mr_ids
        tensor_ptrs = tensor_transport_metadata.tensor_ptrs

        if not mr_ids or not tensor_ptrs:
            self.logger.warning("[UCCL] No tensors to send")
            return

        endpoint = self._get_uccl_endpoint()

        # Get receiver's endpoint info from communicator metadata
        receiver_endpoint_metadata = communicator_metadata.receiver_endpoint_metadata
        receiver_ray_physical_gpu_id = communicator_metadata.receiver_ray_physical_gpu_id

        if not receiver_endpoint_metadata:
            raise RuntimeError("[UCCL] Receiver endpoint metadata not available in communicator metadata")

        # Determine IPC vs RDMA using receiver's metadata
        from uccl.p2p import Endpoint
        import socket

        self.logger.info(f"[UCCL Sender] Parsing receiver metadata: {receiver_endpoint_metadata!r}")
        receiver_ip, receiver_port, receiver_gpu = Endpoint.parse_metadata(receiver_endpoint_metadata)
        self.logger.info(f"[UCCL Sender] Parsed receiver info: IP={receiver_ip}, port={receiver_port}, GPU={receiver_gpu}")

        # Get local info
        local_metadata = endpoint.get_metadata()
        local_ip, _local_port, local_gpu = Endpoint.parse_metadata(local_metadata)
        local_hostname = socket.gethostname()
        receiver_hostname = socket.getfqdn(receiver_ip)

        # Get local Ray physical GPU ID
        local_ray_gpu_ids = ray.get_gpu_ids()
        local_ray_physical_gpu_id = local_ray_gpu_ids[0] if local_ray_gpu_ids else None

        # Determine if IPC (same machine)
        # Use IPC for all same-machine connections to avoid RDMA NIC routing issues
        same_machine = (receiver_ip == local_ip or receiver_ip == "127.0.0.1" or local_hostname == receiver_hostname)
        is_ipc = same_machine

        self.logger.info(f"[UCCL Sender] Using {'IPC' if is_ipc else 'RDMA'} for sending (receiver_ip={receiver_ip}, local_ip={local_ip})")

        transfer_handles = []
        try:
            if is_ipc:
                # IPC: Connect to local receiver (sender connects after receiver is ready)
                self.logger.info(f"[UCCL Sender] Connecting to receiver via IPC on GPU {receiver_gpu} (sender on GPU {local_gpu})...")
                ok, conn_id = endpoint.connect_local(receiver_gpu)
                self.logger.info(f"[UCCL Sender] IPC connect result: ok={ok}, conn_id={conn_id}")
                if not ok:
                    raise RuntimeError(f"[UCCL Sender] Failed to connect via IPC to receiver on GPU {receiver_gpu}")
                self.logger.info(f"[UCCL Sender] IPC connection successful, conn_id={conn_id}")

                # Send tensors via IPC
                for i, tensor in enumerate(tensors):
                    ptr = tensor.data_ptr()
                    size = tensor.numel() * tensor.element_size()
                    self.logger.info(f"[UCCL Sender] Sending tensor {i} via IPC: size={size} bytes")
                    ok, transfer_id = endpoint.send_ipc_async(conn_id, ptr, size)
                    if not ok:
                        raise RuntimeError(f"[UCCL Sender] Failed to initiate IPC send for tensor {i}")
                    transfer_handles.append(transfer_id)
            else:
                # RDMA: Connect to remote receiver (sender connects after receiver is ready)
                if not receiver_port or receiver_port == 0:
                    raise RuntimeError(f"[UCCL Sender] Invalid receiver port: {receiver_port}. Metadata may be corrupted or endpoint not initialized.")
                self.logger.info(f"[UCCL Sender] Connecting to receiver at {receiver_ip}:{receiver_port}, GPU {receiver_gpu}...")
                ok, conn_id = endpoint.connect(receiver_ip, receiver_gpu, remote_port=receiver_port)
                if not ok:
                    raise RuntimeError(f"[UCCL Sender] Failed to connect to receiver at {receiver_ip}:{receiver_port}, GPU {receiver_gpu}")
                self.logger.info(f"[UCCL Sender] RDMA connection successful, conn_id={conn_id}")

                # Send tensors via RDMA using pre-registered memory
                for i, (mr_id, ptr) in enumerate(zip(mr_ids, tensor_ptrs)):
                    size = tensors[i].numel() * tensors[i].element_size()
                    self.logger.info(f"[UCCL Sender] Sending tensor {i} via RDMA: mr_id={mr_id}, size={size} bytes")
                    ok, transfer_id = endpoint.send_async(conn_id, mr_id, ptr, size)
                    if not ok:
                        raise RuntimeError(f"[UCCL Sender] Failed to initiate RDMA send for tensor {i}")
                    transfer_handles.append(transfer_id)

            # Poll all transfers until completion
            for i, transfer_id in enumerate(transfer_handles):
                while True:
                    ok, is_done = endpoint.poll_async(transfer_id)
                    if not ok:
                        raise RuntimeError(f"[UCCL Sender] Polling failed for transfer {i} (id={transfer_id})")
                    if is_done:
                        break
                    time.sleep(0.001)

        except Exception as e:
            self.logger.error(f"[UCCL] Send failed: {e}")
            raise

    def garbage_collect(
        self, obj_id: str, tensor_transport_meta: UCCLTransportMetadata
    ):
        """Clean up resources associated with a tensor transport."""
        from ray._private.worker import global_worker

        gpu_object_store = global_worker.gpu_object_manager.gpu_object_store
        count = gpu_object_store.remove_managed_meta("UCCL", obj_id)

        # Only deregister when no more references
        if count == 0:
            mr_ids = tensor_transport_meta.uccl_mr_ids
            if mr_ids is not None:
                endpoint = self._get_uccl_endpoint()
                for mr_id in mr_ids:
                    endpoint.dereg(mr_id)

    def abort_transport(
        self,
        obj_id: str,
        _communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Abort an ongoing transfer for the given object ID."""
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.add(obj_id)