# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/scheduler_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='protos/scheduler_config.proto',
  package='scheduler.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x1dprotos/scheduler_config.proto\x12\x10scheduler.protos\"\xb7\x01\n\x0fSchedulerConfig\x12\x16\n\x0ehost_addresses\x18\x01 \x03(\t\x12\x17\n\x0finitial_tf_port\x18\x02 \x01(\r\x12\x1e\n\x16num_devices_per_worker\x18\x03 \x01(\r\x12\x1d\n\x15\x65xperiment_time_limit\x18\x04 \x01(\x02\x12\x0b\n\x03ups\x18\x05 \x01(\x02\x12\'\n\x1freorganize_experiments_interval\x18\x06 \x01(\x02')
)




_SCHEDULERCONFIG = _descriptor.Descriptor(
  name='SchedulerConfig',
  full_name='scheduler.protos.SchedulerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='host_addresses', full_name='scheduler.protos.SchedulerConfig.host_addresses', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initial_tf_port', full_name='scheduler.protos.SchedulerConfig.initial_tf_port', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_devices_per_worker', full_name='scheduler.protos.SchedulerConfig.num_devices_per_worker', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='experiment_time_limit', full_name='scheduler.protos.SchedulerConfig.experiment_time_limit', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ups', full_name='scheduler.protos.SchedulerConfig.ups', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reorganize_experiments_interval', full_name='scheduler.protos.SchedulerConfig.reorganize_experiments_interval', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=52,
  serialized_end=235,
)

DESCRIPTOR.message_types_by_name['SchedulerConfig'] = _SCHEDULERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SchedulerConfig = _reflection.GeneratedProtocolMessageType('SchedulerConfig', (_message.Message,), dict(
  DESCRIPTOR = _SCHEDULERCONFIG,
  __module__ = 'protos.scheduler_config_pb2'
  # @@protoc_insertion_point(class_scope:scheduler.protos.SchedulerConfig)
  ))
_sym_db.RegisterMessage(SchedulerConfig)


# @@protoc_insertion_point(module_scope)
