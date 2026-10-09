"""One bounded process-local Fontconfig setup for Cairo/GPU OSD.

No system font install or environment/global-config file mutation. Application
fonts are loaded into a private config and retained by Fontconfig as its current
config. Call at display startup, before any Cairo/GPU font faces are created.
"""
import ctypes as C
from ctypes.util import find_library
from functools import lru_cache
import os
from pathlib import Path

from common.ui_style import FONT_FAMILY, MONO_FAMILY

FONT_DIRECTORY=Path(__file__).resolve().parents[1]/'assets/fonts'


def _match(lib, family):
    pattern=lib.FcNameParse(family.encode())
    if not pattern: raise RuntimeError('font pattern allocation failed')
    matched=None
    try:
        if not lib.FcConfigSubstitute(None,pattern,0): raise RuntimeError('font substitution failed')
        lib.FcDefaultSubstitute(pattern)
        result=C.c_int()
        matched=lib.FcFontMatch(None,pattern,C.byref(result))
        if not matched: raise RuntimeError('bundled font match failed')
        filename=C.c_char_p()
        if lib.FcPatternGetString(matched,b'file',0,C.byref(filename))!=0:
            raise RuntimeError('matched font has no file')
        return Path(os.fsdecode(filename.value)).resolve()
    finally:
        if matched: lib.FcPatternDestroy(matched)
        lib.FcPatternDestroy(pattern)


@lru_cache(maxsize=1)
def configure_fonts():
    lib=C.CDLL(find_library('fontconfig') or 'libfontconfig.so.1')
    signatures={
        'FcConfigCreate':([],C.c_void_p),
        'FcConfigAppFontAddFile':([C.c_void_p,C.c_char_p],C.c_int),
        'FcConfigBuildFonts':([C.c_void_p],C.c_int),
        'FcConfigSetCurrent':([C.c_void_p],C.c_int),
        'FcConfigDestroy':([C.c_void_p],None),
        'FcNameParse':([C.c_char_p],C.c_void_p),
        'FcConfigSubstitute':([C.c_void_p,C.c_void_p,C.c_int],C.c_int),
        'FcDefaultSubstitute':([C.c_void_p],None),
        'FcFontMatch':([C.c_void_p,C.c_void_p,C.POINTER(C.c_int)],C.c_void_p),
        'FcPatternGetString':([C.c_void_p,C.c_char_p,C.c_int,C.POINTER(C.c_char_p)],C.c_int),
        'FcPatternDestroy':([C.c_void_p],None),
    }
    for name,(args,result) in signatures.items():
        function=getattr(lib,name)
        function.argtypes,function.restype=args,result
    config=lib.FcConfigCreate()
    if not config: raise RuntimeError('font configuration allocation failed')
    try:
        for file in ('DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSansMono.ttf'):
            path=FONT_DIRECTORY/file
            if not path.is_file() or not lib.FcConfigAppFontAddFile(config,os.fsencode(path)):
                raise RuntimeError('cannot load bundled font: '+str(path))
        if not lib.FcConfigBuildFonts(config) or not lib.FcConfigSetCurrent(config):
            raise RuntimeError('cannot select process-local font configuration')
        matches={name:_match(lib,family) for name,family in
            (('regular',FONT_FAMILY),('bold',FONT_FAMILY+':style=Bold'),('mono',MONO_FAMILY))}
        for name,file in (('regular','DejaVuSans.ttf'),('bold','DejaVuSans-Bold.ttf'),('mono','DejaVuSansMono.ttf')):
            if matches[name]!=(FONT_DIRECTORY/file).resolve():
                raise RuntimeError('font substitution instead of bundled '+name)
        return matches
    finally:
        # SetCurrent retains its own reference. Release the setup reference,
        # never tear down a config still used by Cairo/GPU font faces.
        lib.FcConfigDestroy(config)
