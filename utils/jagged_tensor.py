import torch
from typing_extensions import Self
from collections.abc import Callable, Generator
from diff_gaussian_rasterization_texture._C import copy_to_resized_tensor, create_jagged_mask

class JaggedTensor:
    def __init__(self, sizes: torch.Tensor | None = None, values: torch.Tensor | None = None, data_dimensionality: int = 3):
        if sizes is None:
            self._sizes: torch.Tensor = torch.empty((0, 2), device="cuda", dtype=torch.int32)
            self._values: torch.Tensor = torch.empty((0, data_dimensionality), device="cuda", dtype=torch.float32)
        else:
            self._sizes: torch.Tensor = sizes
            if values is None:
                # This will give the value 0.1 to the new texture map. Testing with sigmoid activation
                self._values: torch.Tensor = -2.1972 * 0 * torch.ones((int(torch.prod(self._sizes, dim=1).sum().item()), data_dimensionality), device="cuda", dtype=torch.float32)
                # self._values: torch.Tensor = 0.1 * torch.ones((int(torch.prod(self._sizes, dim=1).sum().item()), data_dimensionality), device="cuda", dtype=torch.float32)
            else:
                assert values.shape[0] == torch.prod(self._sizes, dim=1).sum().item() and values.shape[1] == data_dimensionality, "Values and sizes tensors don't match!"
                self._values: torch.Tensor = values

    @property
    # Returns the number of internal tensors
    def n_tensors(self) -> int:
        return self._sizes.shape[0]

    @property
    # Returns the total number of elements per internal tensor
    def n_elements_per_tensor(self) -> torch.Tensor:
        return torch.prod(self._sizes, dim=1)

    @property
    # Returns the total number of elements
    def n_elements(self) -> int:
        return int(self.n_elements_per_tensor.sum().item())

    @property
    # Returns the dimensionality of the values
    def data_dimensionality(self) -> int:
        return self._values.shape[-1]

    @property
    # Offsets in contiguous memory of the starting points of the internal tensors 
    def start_offsets(self) -> torch.Tensor:
        return torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), self.end_offsets[:-1].int()))

    @property
    # Offsets in contiguous memory of the ending points of the internal tensors 
    def end_offsets(self) -> torch.Tensor:
        return torch.cumsum(self.n_elements_per_tensor, dim=0).int()

    # Takes as input either an int/long or a bool mask and converts it to the equivalent int32 mask
    # to be used by the CUDA code
    def _convert_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype == torch.int32 or mask.dtype == torch.int64:
            mask = mask.int()
            if mask.shape[0] == 0:
                raise ValueError()
        elif mask.dtype == torch.bool:
            if mask.sum() == 0:
                raise ValueError()
        else:
            raise TypeError("Invalid mask")
        
        if mask.dtype == torch.bool:
            mask = torch.arange(mask.shape[0], dtype=torch.int32, device="cuda")[mask]
        else:
            mask = mask.view(-1)
        return mask

    # Taking another JaggedTensor as source, copies the internal tensors in a central crop fashion
    # If the sizes of the internal tensors match then it's just a copy
    # If they don't, then the one with dimensions that overflow the target sizes gets cropped
    # to the target size
    # Supports source, target masking as well as recentering of the resulting internal tensor
    # which leads to zero padding for the overflowing elements
    def central_crop(self, 
                     other: Self,
                     source_mask: torch.Tensor | None = None,
                     target_mask: torch.Tensor | None = None,
                     target_center_shift: torch.Tensor | None = None
                     ):
        def check_return_mask(mask: torch.Tensor | None, n_tensors: int) -> torch.Tensor:
            if mask is not None:
                mask = self._convert_mask(mask)
            else:
                mask = torch.arange(n_tensors, device="cuda", dtype=torch.int32)
            return mask

        try:
            source_mask = check_return_mask(source_mask, other.n_tensors)
            target_mask = check_return_mask(target_mask, self.n_tensors)
        except TypeError:
            return JaggedTensor()

        if target_center_shift is not None:
            assert target_center_shift.dim() == 2 and target_center_shift.shape[0] == target_mask.shape[0] and target_center_shift.shape[1] == 2
            target_center_shift = target_center_shift.int()
        else:
            target_center_shift = torch.zeros((target_mask.shape[0], 2), dtype=torch.int32, device="cuda")


        copy_to_resized_tensor(other._values,
                               other._sizes,
                               other.start_offsets,
                               source_mask,
                               target_mask,
                               self._values,
                               self._sizes,
                               self.start_offsets,
                               target_center_shift)
        torch.cuda.empty_cache()


    # Returns the internal tensor at a given index
    def __getitem__(self, idx: int) -> torch.Tensor:
        mask = torch.tensor([idx], device="cuda", dtype=torch.int32)
        jagged_mask = self.create_jagged_mask(mask)
        return self._values[jagged_mask].view(int(self._sizes[idx][0]), int(self._sizes[idx][1]), self.data_dimensionality)

    def mask(self, mask: torch.Tensor) -> Self:
        try:
            mask = self._convert_mask(mask)
        except ValueError:
            return JaggedTensor()
        if mask.dtype == torch.bool:
            source_mask = torch.arange(mask.shape[0], dtype=torch.int32, device="cuda")[mask]
        else:
            source_mask = mask.view(-1)

        jagged_mask = self.create_jagged_mask(source_mask)

        target_sizes = self._sizes[mask]
        target_values = self._values[jagged_mask]
        target_jagged_tensor = JaggedTensor(target_sizes, target_values)

        return target_jagged_tensor

    # Creates a mask for the jagged, values tensor
    # from a mask over the internal tensors
    def create_jagged_mask(self, mask: torch.Tensor) -> torch.Tensor:
        jagged_mask: torch.Tensor = torch.zeros(self.n_elements, dtype=torch.bool, device="cuda")
        if mask.dtype == torch.int32 or mask.dtype == torch.int64:
            boolean_mask = torch.zeros(self.n_tensors, device="cuda", dtype=torch.bool)
            boolean_mask[mask] = True
        else:
            assert mask.shape[0] == self.n_tensors, "Mask for jagged mask creation has wrong number of elements!"
            boolean_mask = mask
        
        create_jagged_mask(boolean_mask, jagged_mask, self._sizes, self.start_offsets)

        return jagged_mask

    def _iter(self) -> Generator[torch.Tensor, torch.Tensor, None]:
        for curr_tex_res in self._sizes.unique(dim=0):
            boolean_mask = (self._sizes == curr_tex_res).all(dim=1)
            yield curr_tex_res, boolean_mask

    # Function that takes as input either a JaggedTensor of the exact same sizes and adds the corresponding elements
    # or a normal tensor with dimensions (n_internal_tensors, data_dimensionality)
    # and for each element, it adds it to the entire, corresponding interal tensor
    # TODO: Optionally give an activation function. Maybe add it to the whole class? I dunno
    def add(self, other: torch.Tensor | Self, activation_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
        cloned_tensor = self.clone()
        if isinstance(other, torch.Tensor):
            assert other.dim() == 2 and other.shape[0] == self.n_tensors and other.shape[1] == self.data_dimensionality, "Invalid tensor to add!"
            cloned_tensor._values = activation_fn(cloned_tensor._values) + other.repeat_interleave(repeats=torch.prod(self._sizes, dim=1).int(), dim=0)
        elif isinstance(other, JaggedTensor):
            assert other.data_dimensionality == self.data_dimensionality and (other._sizes == self._sizes).all(), "Jagged Tensors' dimensions don't match!"
            cloned_tensor._values = other._values + cloned_tensor._values
        return cloned_tensor

    # Function that takes as input either a JaggedTensor of the exact same sizes and adds the corresponding elements
    # or a normal tensor with dimensions (n_internal_tensors, data_dimensionality)
    # and for each element, it adds it to the entire, corresponding interal tensor
    def multiply(self, other: torch.Tensor | Self):
        cloned_tensor = self.clone()
        if isinstance(other, torch.Tensor):
            assert other.dim() == 2 and other.shape[0] == self.n_tensors and (other.shape[1] == self.data_dimensionality or other.shape[1] == 1), "Invalid tensor to multiply!"
            for curr_tex_res, boolean_mask in cloned_tensor._iter():
                jagged_mask = self.create_jagged_mask(boolean_mask)
                cloned_tensor._values[jagged_mask] = (other[boolean_mask].view(int(boolean_mask.sum()), 1, 1, -1) * cloned_tensor._values[jagged_mask].view(-1, curr_tex_res[0], curr_tex_res[1], self.data_dimensionality)).view(-1, self.data_dimensionality)
        elif isinstance(other, JaggedTensor):
            assert other.data_dimensionality == self.data_dimensionality and (other._sizes == self._sizes).all(), "Jagged Tensors' dimensions don't match!"
            cloned_tensor._values = other._values * cloned_tensor._values
        return cloned_tensor

    # Returns a copy of the JaggedTensor
    def clone(self) -> Self:
        new_jagged = JaggedTensor(self._sizes.clone(), self._values.clone())
        return new_jagged
    
    # TODO The activation function shouldn't be necessary (helpful for debugging, but not necessary)
    def generate_downscaled_reconstructed_maps(self, mask: torch.Tensor, activation_fn: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            mask = self._convert_mask(mask)
        except ValueError:
            return (torch.empty(device="cuda"), torch.empty(device="cuda"), torch.empty(device="cuda"))

        if self._sizes[mask].unique(dim=0).shape[0] != 1:
            raise ValueError("The masked elements do not have the same sizes. Cannot operate interpolation")

        if mask.dtype == torch.bool:
            source_mask = torch.arange(mask.shape[0], dtype=torch.int32, device="cuda")[mask]
        else:
            source_mask = mask.view(-1)

        jagged_mask = self.create_jagged_mask(source_mask)

        curr_sizes: tuple[int, int] = int(self._sizes[mask[0]][0]), int(self._sizes[mask[0]][1])

        activated_values = activation_fn(self._values)[jagged_mask].view(-1, curr_sizes[0], curr_sizes[1], 3)

        original = activated_values.permute(0, 3, 1, 2)
        downscaled = torch.nn.functional.interpolate(original, mode="bicubic", antialias=True, align_corners=True, size=(curr_sizes[0]//2, curr_sizes[1]//2))
        reconstructed = torch.nn.functional.interpolate(downscaled, mode="bicubic", antialias=True, align_corners=True, size=(curr_sizes[0], curr_sizes[1])).permute(0, 2, 3, 1)
        original = original.permute(0, 2, 3, 1)
        downscaled = downscaled.permute(0, 2, 3, 1)
        return original, downscaled, reconstructed
    
