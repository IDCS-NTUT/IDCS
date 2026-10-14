"""Enumerate both forms of GStreamer pad data without importing GI."""


def probe_buffers(info, *, buffer_flag, buffer_list_flag):
    if info.type & buffer_list_flag:
        buffers = info.get_buffer_list()
        if buffers is not None:
            for index in range(buffers.length()):
                buffer = buffers.get(index)
                if buffer is not None:
                    yield buffer
    elif info.type & buffer_flag:
        buffer = info.get_buffer()
        if buffer is not None:
            yield buffer
