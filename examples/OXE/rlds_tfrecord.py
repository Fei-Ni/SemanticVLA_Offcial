from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _build_example_cls():
    fd = descriptor_pb2.FileDescriptorProto(name="tf_example_minimal.proto", package="tf")

    def add_msg(name: str):
        msg = fd.message_type.add()
        msg.name = name
        return msg

    bytes_list = add_msg("BytesList")
    field = bytes_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    float_list = add_msg("FloatList")
    field = float_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    field.options.packed = True

    int64_list = add_msg("Int64List")
    field = int64_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    field.options.packed = True

    feature = add_msg("Feature")
    oneof = feature.oneof_decl.add()
    oneof.name = "kind"
    for name, number, type_name in (
        ("bytes_list", 1, ".tf.BytesList"),
        ("float_list", 2, ".tf.FloatList"),
        ("int64_list", 3, ".tf.Int64List"),
    ):
        field = feature.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = type_name
        field.oneof_index = 0

    features = add_msg("Features")
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    field = entry.field.add()
    field.name = "key"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    field = entry.field.add()
    field.name = "value"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tf.Feature"
    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tf.Features.FeatureEntry"

    example = add_msg("Example")
    field = example.field.add()
    field.name = "features"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tf.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("tf.Example"))


_EXAMPLE_CLS = _build_example_cls()


def read_record_at(path: str | Path, record_index: int) -> bytes:
    """Read one serialized Example from an uncompressed TFRecord file."""
    target = int(record_index)
    with open(path, "rb") as fp:
        for idx in range(target + 1):
            header = fp.read(8)
            if not header:
                raise IndexError(f"record {target} out of range in {path}")
            if len(header) != 8:
                raise IOError(f"truncated TFRecord length header in {path}")
            length = struct.unpack("<Q", header)[0]
            fp.read(4)  # masked crc32c for length
            data = fp.read(length)
            if len(data) != length:
                raise IOError(f"truncated TFRecord payload in {path}")
            fp.read(4)  # masked crc32c for payload
            if idx == target:
                return data
    raise IndexError(f"record {target} out of range in {path}")


def iter_records(path: str | Path) -> Iterator[bytes]:
    """Yield serialized Examples from an uncompressed TFRecord file."""
    with open(path, "rb") as fp:
        while True:
            header = fp.read(8)
            if not header:
                return
            if len(header) != 8:
                raise IOError(f"truncated TFRecord length header in {path}")
            length = struct.unpack("<Q", header)[0]
            fp.read(4)
            data = fp.read(length)
            if len(data) != length:
                raise IOError(f"truncated TFRecord payload in {path}")
            fp.read(4)
            yield data


def parse_example(serialized: bytes) -> Any:
    example = _EXAMPLE_CLS()
    example.ParseFromString(serialized)
    return example


def bytes_feature(example: Any, key: str) -> list[bytes]:
    feat = example.features.feature[key]
    return list(feat.bytes_list.value)


def float_feature(example: Any, key: str) -> list[float]:
    feat = example.features.feature[key]
    return list(feat.float_list.value)


def int64_feature(example: Any, key: str) -> list[int]:
    feat = example.features.feature[key]
    return list(feat.int64_list.value)
