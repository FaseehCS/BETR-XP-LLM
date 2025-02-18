from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PingRequest(_message.Message):
    __slots__ = ["seq"]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    seq: int
    def __init__(self, seq: _Optional[int] = ...) -> None: ...

class PingReply(_message.Message):
    __slots__ = ["seq"]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    seq: int
    def __init__(self, seq: _Optional[int] = ...) -> None: ...

class Mask(_message.Message):
    __slots__ = ["w", "h", "score", "packedbits"]
    W_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    PACKEDBITS_FIELD_NUMBER: _ClassVar[int]
    w: int
    h: int
    score: float
    packedbits: bytes
    def __init__(self, w: _Optional[int] = ..., h: _Optional[int] = ..., score: _Optional[float] = ..., packedbits: _Optional[bytes] = ...) -> None: ...

class Region(_message.Message):
    __slots__ = ["x", "y", "w", "h"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    H_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    w: int
    h: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ..., w: _Optional[int] = ..., h: _Optional[int] = ...) -> None: ...

class Image(_message.Message):
    __slots__ = ["image_format", "image_data"]
    IMAGE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    image_format: str
    image_data: bytes
    def __init__(self, image_format: _Optional[str] = ..., image_data: _Optional[bytes] = ...) -> None: ...

class Pose(_message.Message):
    __slots__ = ["position", "orientation"]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    position: _containers.RepeatedScalarFieldContainer[float]
    orientation: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, position: _Optional[_Iterable[float]] = ..., orientation: _Optional[_Iterable[float]] = ...) -> None: ...

class XYZ(_message.Message):
    __slots__ = ["xyz"]
    XYZ_FIELD_NUMBER: _ClassVar[int]
    xyz: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, xyz: _Optional[_Iterable[float]] = ...) -> None: ...

class Points(_message.Message):
    __slots__ = ["points"]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[XYZ]
    def __init__(self, points: _Optional[_Iterable[_Union[XYZ, _Mapping]]] = ...) -> None: ...

class PromptObjectDetectionRequest(_message.Message):
    __slots__ = ["api_key", "prompt", "image"]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    prompt: str
    image: Image
    def __init__(self, api_key: _Optional[str] = ..., prompt: _Optional[str] = ..., image: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class ObjectDetectionRequest(_message.Message):
    __slots__ = ["api_key", "image"]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    image: Image
    def __init__(self, api_key: _Optional[str] = ..., image: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class ObjectDetectionReply(_message.Message):
    __slots__ = ["masks", "regions", "label"]
    MASKS_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    masks: _containers.RepeatedCompositeFieldContainer[Mask]
    regions: _containers.RepeatedCompositeFieldContainer[Region]
    label: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, masks: _Optional[_Iterable[_Union[Mask, _Mapping]]] = ..., regions: _Optional[_Iterable[_Union[Region, _Mapping]]] = ..., label: _Optional[_Iterable[str]] = ...) -> None: ...

class PointPoseDetectionRequest(_message.Message):
    __slots__ = ["api_key", "prompt", "rgb", "points", "box_threshold"]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    RGB_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    BOX_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    prompt: str
    rgb: Image
    points: Points
    box_threshold: float
    def __init__(self, api_key: _Optional[str] = ..., prompt: _Optional[str] = ..., rgb: _Optional[_Union[Image, _Mapping]] = ..., points: _Optional[_Union[Points, _Mapping]] = ..., box_threshold: _Optional[float] = ...) -> None: ...

class PoseDetectionRequest(_message.Message):
    __slots__ = ["api_key", "prompt", "rgb", "depth", "intrinsics", "box_threshold"]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    RGB_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    BOX_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    prompt: str
    rgb: Image
    depth: Image
    intrinsics: _containers.RepeatedScalarFieldContainer[float]
    box_threshold: float
    def __init__(self, api_key: _Optional[str] = ..., prompt: _Optional[str] = ..., rgb: _Optional[_Union[Image, _Mapping]] = ..., depth: _Optional[_Union[Image, _Mapping]] = ..., intrinsics: _Optional[_Iterable[float]] = ..., box_threshold: _Optional[float] = ...) -> None: ...

class PoseDetectionReply(_message.Message):
    __slots__ = ["masks", "regions", "label", "pose"]
    MASKS_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    masks: _containers.RepeatedCompositeFieldContainer[Mask]
    regions: _containers.RepeatedCompositeFieldContainer[Region]
    label: _containers.RepeatedScalarFieldContainer[str]
    pose: _containers.RepeatedCompositeFieldContainer[Pose]
    def __init__(self, masks: _Optional[_Iterable[_Union[Mask, _Mapping]]] = ..., regions: _Optional[_Iterable[_Union[Region, _Mapping]]] = ..., label: _Optional[_Iterable[str]] = ..., pose: _Optional[_Iterable[_Union[Pose, _Mapping]]] = ...) -> None: ...