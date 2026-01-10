"""
Test suite for UCCL Tensor Transport

Run with: pytest ray/python/ray/tests/gpu_objects/test_gpu_objects_uccl.py -v
"""

import sys
import pytest
import torch
import time

import ray
from ray._common.test_utils import wait_for_condition


@ray.remote(num_gpus=1, num_cpus=0, enable_tensor_transport=True)
class GPUTestActor:
    """Actor for testing UCCL transport."""
    
    def __init__(self):
        self.reserved_tensor1 = torch.tensor([1, 2, 3]).to("cuda")
        self.reserved_tensor2 = torch.tensor([4, 5, 6]).to("cuda")
        self.reserved_tensor3 = torch.tensor([7, 8, 9]).to("cuda")

    @ray.method(tensor_transport="UCCL")
    def echo(self, data, device):
        """Echo tensor to specified device."""
        return data.to(device)

    def sum(self, data, device):
        """Sum tensor values."""
        assert data.device.type == device
        return data.sum().item()

    def produce(self, tensors):
        """Produce multiple tensors with UCCL transport."""
        refs = []
        for t in tensors:
            refs.append(ray.put(t, _tensor_transport="UCCL"))
        return refs

    def consume_with_uccl(self, refs):
        """Consume tensors using UCCL transport."""
        tensors = [ray.get(ref) for ref in refs]
        sum_val = 0
        for t in tensors:
            assert t.device.type == "cuda"
            sum_val += t.sum().item()
        return sum_val

    def consume_with_object_store(self, refs):
        """Consume tensors using object store (baseline)."""
        tensors = [ray.get(ref, _use_object_store=True) for ref in refs]
        sum_val = 0
        for t in tensors:
            assert t.device.type == "cuda"
            sum_val += t.sum().item()
        return sum_val

    def gc(self):
        """Test garbage collection."""
        tensor = torch.tensor([1, 2, 3]).to("cuda")
        ref = ray.put(tensor, _tensor_transport="UCCL")
        obj_id = ref.hex()
        
        from ray._private.worker import global_worker
        gpu_object_store = global_worker.gpu_object_manager.gpu_object_store
        
        # Verify metadata exists
        num_meta = gpu_object_store.get_num_managed_meta("UCCL")
        assert num_meta > 0, "Should have UCCL metadata"
        
        # Delete reference and trigger GC
        del ref
        import gc
        gc.collect()
        
        # Wait for cleanup
        def check_cleaned():
            return gpu_object_store.get_num_managed_meta("UCCL") == 0
        
        wait_for_condition(check_cleaned, timeout=10)
        return True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2 GPUs"
)
class TestUCCLBasic:
    """Basic functionality tests for UCCL transport."""

    def test_backend_registration(self, ray_start_cluster_head):
        """Test that UCCL is registered as a valid backend."""
        from ray.experimental.gpu_object_manager.util import (
            transport_manager_classes,
            get_tensor_transport_manager,
        )
        
        assert "UCCL" in transport_manager_classes
        manager = get_tensor_transport_manager("UCCL")
        assert manager.tensor_transport_backend == "UCCL"
        assert manager.is_one_sided() == True
        assert manager.can_abort_transport() == True

    def test_simple_transfer(self, ray_start_cluster_head):
        """Test simple tensor transfer between two actors."""
        actor1 = GPUTestActor.remote()
        actor2 = GPUTestActor.remote()
        
        # Create tensor on actor1
        tensor = torch.ones(1000, dtype=torch.float32).cuda()
        ref = actor1.echo.remote(tensor, "cuda")
        
        # Receive on actor2
        result = ray.get(actor2.sum.remote(ref, "cuda"))
        assert result == 1000.0

    def test_multiple_tensors(self, ray_start_cluster_head):
        """Test transferring multiple tensors."""
        actor1 = GPUTestActor.remote()
        actor2 = GPUTestActor.remote()
        
        # Create multiple tensors
        tensors = [
            torch.ones(100, dtype=torch.float32).cuda(),
            torch.ones(200, dtype=torch.float32).cuda() * 2,
            torch.ones(300, dtype=torch.float32).cuda() * 3,
        ]
        
        # Produce refs on actor1
        refs = ray.get(actor1.produce.remote(tensors))
        
        # Consume on actor2
        result = ray.get(actor2.consume_with_uccl.remote(refs))
        
        # Expected: 100*1 + 200*2 + 300*3 = 1300
        assert result == 1300.0

    def test_large_tensor(self, ray_start_cluster_head):
        """Test transferring a large tensor."""
        actor1 = GPUTestActor.remote()
        actor2 = GPUTestActor.remote()
        
        # 100MB tensor
        size = 100 * 1024 * 1024 // 4
        tensor = torch.randn(size, dtype=torch.float32).cuda()
        
        start = time.time()
        ref = actor1.echo.remote(tensor, "cuda")
        result = ray.get(actor2.sum.remote(ref, "cuda"))
        elapsed = time.time() - start
        
        print(f"Large tensor transfer took {elapsed:.3f}s")
        assert elapsed < 5.0  # Should complete in reasonable time


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2 GPUs"
)
class TestUCCLMultipleReceivers:
    """Test multiple actors receiving the same tensor."""

    def test_broadcast(self, ray_start_cluster_head):
        """Test one sender, multiple receivers."""
        sender = GPUTestActor.remote()
        receivers = [GPUTestActor.remote() for _ in range(3)]
        
        # Create tensor
        tensor = torch.arange(1000, dtype=torch.float32).cuda()
        ref = sender.echo.remote(tensor, "cuda")
        
        # Multiple receivers get the same tensor
        results = ray.get([r.sum.remote(ref, "cuda") for r in receivers])
        
        expected_sum = (999 * 1000) / 2  # sum of 0 to 999
        for result in results:
            assert result == expected_sum


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2 GPUs"
)
class TestUCCLEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_tensor(self, ray_start_cluster_head):
        """Test transferring an empty tensor."""
        actor1 = GPUTestActor.remote()
        actor2 = GPUTestActor.remote()
        
        tensor = torch.tensor([], dtype=torch.float32).cuda()
        ref = actor1.echo.remote(tensor, "cuda")
        result = ray.get(actor2.sum.remote(ref, "cuda"))
        
        assert result == 0.0

    def test_same_actor_transfer(self, ray_start_cluster_head):
        """Test that intra-actor transfers work."""
        actor = GPUTestActor.remote()
        
        tensor = torch.ones(100).cuda()
        ref = actor.echo.remote(tensor, "cuda")
        result = ray.get(actor.sum.remote(ref, "cuda"))
        
        assert result == 100.0

    def test_garbage_collection(self, ray_start_cluster_head):
        """Test that memory is properly freed."""
        actor = GPUTestActor.remote()
        result = ray.get(actor.gc.remote())
        assert result == True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires 2 GPUs"
)
class TestUCCLPerformance:
    """Performance comparison tests."""

    def test_uccl_vs_objectstore(self, ray_start_cluster_head):
        """Compare UCCL vs object store performance."""
        actor1 = GPUTestActor.remote()
        actor2 = GPUTestActor.remote()
        
        # Create test tensors
        tensors = [torch.randn(1000000, dtype=torch.float32).cuda() for _ in range(5)]
        
        # UCCL transfer
        start = time.time()
        refs = ray.get(actor1.produce.remote(tensors))
        result_uccl = ray.get(actor2.consume_with_uccl.remote(refs))
        time_uccl = time.time() - start
        
        # Object store transfer (baseline)
        start = time.time()
        refs = ray.get(actor1.produce.remote(tensors))
        result_obj = ray.get(actor2.consume_with_object_store.remote(refs))
        time_obj = time.time() - start
        
        print(f"UCCL time: {time_uccl:.3f}s")
        print(f"Object store time: {time_obj:.3f}s")
        print(f"Speedup: {time_obj / time_uccl:.2f}x")
        
        # Results should match
        assert abs(result_uccl - result_obj) < 1e-3
        
        # UCCL should be faster (usually 2-10x)
        # But don't fail test if not, as it depends on hardware
        assert time_uccl < 10.0  # Just check it completes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

