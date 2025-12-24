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

if TYPE_CHECKING:
    import torch


@dataclass
class UCCLCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the UCCL communicator."""
    pass


@dataclass
class UCCLTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for UCCL transport.

    Args:
        uccl_endpoint_metadata: Serialized endpoint metadata (IP, port, GPU index).
        uccl_mr_ids: Memory region IDs for registered tensors.
        tensor_ptrs: Data pointers for the tensors.
    """
    uccl_endpoint_metadata: Optional[bytes] = None
    uccl_mr_ids: Optional[List[int]] = None
    tensor_ptrs: Optional[List[int]] = None

    __eq__ = object.__eq__
    __hash__ = object.__hash__

class UCCLTensorTransport(TensorTransportManager):
    def __init__(self, tensor_transport_backend: str):
        # Lazy initialization to avoid importing UCCL unnecessarily
        self._uccl_endpoint = None
        self._aborted_transfer_obj_ids = set()
        self._aborted_transfer_obj_ids_lock = threading.Lock()
        # Cache for established connections: (remote_endpoint_metadata) -> conn_id
        self._connections = {}
        self._connections_lock = threading.Lock()

    @property
    def tensor_transport_backend(self) -> str:
        return "UCCL"

    @staticmethod
    def is_one_sided() -> bool:
        # UCCL supports dual-sided transport, but let us start from supporting one-sided transport first
        return True

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
        # TODO: check how to get these info from the runtime context
        local_gpu_idx = ray.get_gpu_ids()[0] if ray.get_gpu_ids() else 0
        # Use a reasonable default for num_cpus
        num_cpus = 4
        self._uccl_endpoint = Endpoint(local_gpu_idx, num_cpus)
        return self._uccl_endpoint

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        """Check if the remote actor has UCCL tensor transport available."""
        def __ray_actor_has_tensor_transport__(
            self: "ray.actor.ActorHandle",
        ) -> bool:
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
        )
        # Cache the metadata using the generic API
        gpu_object_store.record_managed_meta("UCCL", obj_id, ret)
        return ret

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> UCCLCommunicatorMetadata:
        """Get communicator metadata for UCCL transport."""
        return UCCLCommunicatorMetadata()

    def _get_or_create_connection(self, remote_endpoint_metadata: bytes) -> int:
        """Get existing connection or create a new one to the remote endpoint."""
        # Use metadata as connection key
        with self._connections_lock:
            if remote_endpoint_metadata in self._connections:
                return self._connections[remote_endpoint_metadata]

            # Parse remote endpoint info
            from uccl.p2p import Endpoint
            ip, port, remote_gpu = Endpoint.parse_metadata(remote_endpoint_metadata)

            # Connect to the remote endpoint
            endpoint = self._get_uccl_endpoint()
            ok, conn_id = endpoint.connect(ip, remote_gpu, remote_port=port)
            if not ok:
                raise RuntimeError(
                    f"Failed to connect to remote endpoint at {ip}:{port} (GPU {remote_gpu})"
                )

            # Cache the connection
            self._connections[remote_endpoint_metadata] = conn_id
            return conn_id

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

        remote_endpoint_metadata = tensor_transport_metadata.uccl_endpoint_metadata
        remote_mr_ids = tensor_transport_metadata.uccl_mr_ids
        remote_tensor_ptrs = tensor_transport_metadata.tensor_ptrs

        # Check if transfer was aborted before starting
        with self._aborted_transfer_obj_ids_lock:
            if obj_id in self._aborted_transfer_obj_ids:
                self._aborted_transfer_obj_ids.remove(obj_id)
                raise RuntimeError(f"UCCL transfer aborted for object id: {obj_id}")

        endpoint = self._get_uccl_endpoint()
        local_mr_ids = []
        transfer_handles = []

        try:
            # Get or create connection to the remote endpoint
            conn_id = self._get_or_create_connection(remote_endpoint_metadata)

            # Register local tensors for receiving
            for tensor in tensors:
                ptr = tensor.data_ptr()
                size = tensor.numel() * tensor.element_size()
                ok, mr_id = endpoint.reg(ptr, size)
                if not ok:
                    raise RuntimeError("Failed to register local memory for receiving")
                local_mr_ids.append((mr_id, ptr, size))

            # Initiate async receives for all tensors
            for i, (local_mr_id, local_ptr, size) in enumerate(local_mr_ids):
                ok, transfer_id = endpoint.recv_async(
                    conn_id, local_mr_id, local_ptr, size
                )
                if not ok:
                    raise RuntimeError(f"Failed to initiate async recv for tensor {i}")
                transfer_handles.append(transfer_id)

            # Poll all transfers until completion
            for transfer_id in transfer_handles:
                while True:
                    # Check for abort
                    with self._aborted_transfer_obj_ids_lock:
                        if obj_id in self._aborted_transfer_obj_ids:
                            self._aborted_transfer_obj_ids.remove(obj_id)
                            raise RuntimeError(
                                f"UCCL transfer aborted for object id: {obj_id}"
                            )

                    ok, is_done = endpoint.poll_async(transfer_id)
                    if not ok:
                        raise RuntimeError("Error polling async transfer")

                    if is_done:
                        break

                    time.sleep(0.001)  # Avoid busy waiting

        finally:
            # Best effort cleanup
            with self._aborted_transfer_obj_ids_lock:
                self._aborted_transfer_obj_ids.discard(obj_id)

            # Deregister local memory regions
            for mr_id, _, _ in local_mr_ids:
                endpoint.dereg(mr_id)

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: UCCLTransportMetadata,
        communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """UCCL is configured as one-sided transport, so send is not used."""
        raise NotImplementedError(
            "UCCL transport does not support send_multiple_tensors, since it is a one-sided transport."
        )

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
        communicator_metadata: UCCLCommunicatorMetadata,
    ):
        """Abort an ongoing transfer for the given object ID."""
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.add(obj_id)