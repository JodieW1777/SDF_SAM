from pydantic import BaseModel, Field, model_validator


class ReconstructionOptions(BaseModel):
    bbox_index: list[float] = Field(
        ..., min_length=6, max_length=6,
        description="MITK continuous-index bbox [x0,y0,z0,x1,y1,z1]",
    )
    index_to_world: list[float] = Field(
        ...,
        min_length=16,
        max_length=16,
        description="Row-major 4x4 MITK index-to-world matrix",
    )
    num_slices: int = Field(8, ge=4, le=32)
    query_budget: int = Field(100_000, ge=8_000, le=2_000_000)
    query_chunk_size: int = Field(20_000, ge=512, le=100_000)
    level: float = 0.0

    @model_validator(mode="after")
    def validate_bbox(self):
        lo = self.bbox_index[:3]
        hi = self.bbox_index[3:]
        if any(b <= a for a, b in zip(lo, hi)):
            raise ValueError("bbox upper bounds must be greater than lower bounds")
        return self
