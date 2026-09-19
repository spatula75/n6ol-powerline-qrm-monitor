"""ctypes declarations for the SDRplay API.  GENERATED FILE - do not edit.

tools/generate_sdrplay_api.py wrote this from the SDRplay API version 3.15
headers in sdrplay-api-3.15.  Edit the generator or install a newer API and
regenerate, because the next run discards whatever anybody changes here.

Nothing in here talks to a device.  This module states the shapes the SDRplay
library expects, so that a device shim can call that library.  One wrong field is
memory corruption rather than an exception, so these declarations are generated
rather than transcribed by hand.
"""
import ctypes
from enum import IntEnum

API_VERSION = 3.15

# The headers define these constants.
MAX_BB_GR = 59
RSPIA_NUM_LNA_STATES = 10
RSPIA_NUM_LNA_STATES_AM = 7
RSPIA_NUM_LNA_STATES_LBAND = 9
RSPII_NUM_LNA_STATES = 9
RSPII_NUM_LNA_STATES_AMPORT = 5
RSPII_NUM_LNA_STATES_420MHZ = 6
RSPDUO_NUM_LNA_STATES = 10
RSPDUO_NUM_LNA_STATES_AMPORT = 5
RSPDUO_NUM_LNA_STATES_AM = 7
RSPDUO_NUM_LNA_STATES_LBAND = 9
RSPDX_NUM_LNA_STATES = 28
RSPDX_NUM_LNA_STATES_AMPORT2_0_12 = 19
RSPDX_NUM_LNA_STATES_AMPORT2_12_50 = 20
RSPDX_NUM_LNA_STATES_AMPORT2_50_60 = 25
RSPDX_NUM_LNA_STATES_VHF_BAND3 = 27
RSPDX_NUM_LNA_STATES_420MHZ = 21
RSPDX_NUM_LNA_STATES_LBAND = 19
RSPDX_NUM_LNA_STATES_DX = 22
SDRPLAY_MAX_DEVICES = 16
SDRPLAY_MAX_TUNERS_PER_DEVICE = 2
SDRPLAY_MAX_SER_NO_LEN = 64
SDRPLAY_MAX_ROOT_NM_LEN = 32
SDRPLAY_RSP1_ID = 1
SDRPLAY_RSP1A_ID = 255
SDRPLAY_RSP2_ID = 2
SDRPLAY_RSPduo_ID = 3
SDRPLAY_RSPdx_ID = 4
SDRPLAY_RSP1B_ID = 6
SDRPLAY_RSPdxR2_ID = 7


class sdrplay_api_Bw_MHzT(IntEnum):
    sdrplay_api_BW_Undefined = 0
    sdrplay_api_BW_0_200 = 200
    sdrplay_api_BW_0_300 = 300
    sdrplay_api_BW_0_600 = 600
    sdrplay_api_BW_1_536 = 1536
    sdrplay_api_BW_5_000 = 5000
    sdrplay_api_BW_6_000 = 6000
    sdrplay_api_BW_7_000 = 7000
    sdrplay_api_BW_8_000 = 8000


class sdrplay_api_If_kHzT(IntEnum):
    sdrplay_api_IF_Undefined = -1
    sdrplay_api_IF_Zero = 0
    sdrplay_api_IF_0_450 = 450
    sdrplay_api_IF_1_620 = 1620
    sdrplay_api_IF_2_048 = 2048


class sdrplay_api_LoModeT(IntEnum):
    sdrplay_api_LO_Undefined = 0
    sdrplay_api_LO_Auto = 1
    sdrplay_api_LO_120MHz = 2
    sdrplay_api_LO_144MHz = 3
    sdrplay_api_LO_168MHz = 4


class sdrplay_api_MinGainReductionT(IntEnum):
    sdrplay_api_EXTENDED_MIN_GR = 0
    sdrplay_api_NORMAL_MIN_GR = 20


class sdrplay_api_TunerSelectT(IntEnum):
    sdrplay_api_Tuner_Neither = 0
    sdrplay_api_Tuner_A = 1
    sdrplay_api_Tuner_B = 2
    sdrplay_api_Tuner_Both = 3


class sdrplay_api_GainValuesT(ctypes.Structure):
    _fields_ = [
        ('curr', ctypes.c_float),
        ('max', ctypes.c_float),
        ('min', ctypes.c_float),
    ]


class sdrplay_api_GainT(ctypes.Structure):
    _fields_ = [
        ('gRdB', ctypes.c_int),
        ('LNAstate', ctypes.c_ubyte),
        ('syncUpdate', ctypes.c_ubyte),
        ('minGr', ctypes.c_int),
        ('gainVals', sdrplay_api_GainValuesT),
    ]


class sdrplay_api_RfFreqT(ctypes.Structure):
    _fields_ = [
        ('rfHz', ctypes.c_double),
        ('syncUpdate', ctypes.c_ubyte),
    ]


class sdrplay_api_DcOffsetTunerT(ctypes.Structure):
    _fields_ = [
        ('dcCal', ctypes.c_ubyte),
        ('speedUp', ctypes.c_ubyte),
        ('trackTime', ctypes.c_int),
        ('refreshRateTime', ctypes.c_int),
    ]


class sdrplay_api_TunerParamsT(ctypes.Structure):
    _fields_ = [
        ('bwType', ctypes.c_int),
        ('ifType', ctypes.c_int),
        ('loMode', ctypes.c_int),
        ('gain', sdrplay_api_GainT),
        ('rfFreq', sdrplay_api_RfFreqT),
        ('dcOffsetTuner', sdrplay_api_DcOffsetTunerT),
    ]


class sdrplay_api_AgcControlT(IntEnum):
    sdrplay_api_AGC_DISABLE = 0
    sdrplay_api_AGC_100HZ = 1
    sdrplay_api_AGC_50HZ = 2
    sdrplay_api_AGC_5HZ = 3
    sdrplay_api_AGC_CTRL_EN = 4


class sdrplay_api_AdsbModeT(IntEnum):
    sdrplay_api_ADSB_DECIMATION = 0
    sdrplay_api_ADSB_NO_DECIMATION_LOWPASS = 1
    sdrplay_api_ADSB_NO_DECIMATION_BANDPASS_2MHZ = 2
    sdrplay_api_ADSB_NO_DECIMATION_BANDPASS_3MHZ = 3


class sdrplay_api_DcOffsetT(ctypes.Structure):
    _fields_ = [
        ('DCenable', ctypes.c_ubyte),
        ('IQenable', ctypes.c_ubyte),
    ]


class sdrplay_api_DecimationT(ctypes.Structure):
    _fields_ = [
        ('enable', ctypes.c_ubyte),
        ('decimationFactor', ctypes.c_ubyte),
        ('wideBandSignal', ctypes.c_ubyte),
    ]


class sdrplay_api_AgcT(ctypes.Structure):
    _fields_ = [
        ('enable', ctypes.c_int),
        ('setPoint_dBfs', ctypes.c_int),
        ('attack_ms', ctypes.c_ushort),
        ('decay_ms', ctypes.c_ushort),
        ('decay_delay_ms', ctypes.c_ushort),
        ('decay_threshold_dB', ctypes.c_ushort),
        ('syncUpdate', ctypes.c_int),
    ]


class sdrplay_api_ControlParamsT(ctypes.Structure):
    _fields_ = [
        ('dcOffset', sdrplay_api_DcOffsetT),
        ('decimation', sdrplay_api_DecimationT),
        ('agc', sdrplay_api_AgcT),
        ('adsbMode', ctypes.c_int),
    ]


class sdrplay_api_Rsp1aParamsT(ctypes.Structure):
    _fields_ = [
        ('rfNotchEnable', ctypes.c_ubyte),
        ('rfDabNotchEnable', ctypes.c_ubyte),
    ]


class sdrplay_api_Rsp1aTunerParamsT(ctypes.Structure):
    _fields_ = [
        ('biasTEnable', ctypes.c_ubyte),
    ]


class sdrplay_api_Rsp2_AntennaSelectT(IntEnum):
    sdrplay_api_Rsp2_ANTENNA_A = 5
    sdrplay_api_Rsp2_ANTENNA_B = 6


class sdrplay_api_Rsp2_AmPortSelectT(IntEnum):
    sdrplay_api_Rsp2_AMPORT_1 = 1
    sdrplay_api_Rsp2_AMPORT_2 = 0


class sdrplay_api_Rsp2ParamsT(ctypes.Structure):
    _fields_ = [
        ('extRefOutputEn', ctypes.c_ubyte),
    ]


class sdrplay_api_Rsp2TunerParamsT(ctypes.Structure):
    _fields_ = [
        ('biasTEnable', ctypes.c_ubyte),
        ('amPortSel', ctypes.c_int),
        ('antennaSel', ctypes.c_int),
        ('rfNotchEnable', ctypes.c_ubyte),
    ]


class sdrplay_api_RspDuoModeT(IntEnum):
    sdrplay_api_RspDuoMode_Unknown = 0
    sdrplay_api_RspDuoMode_Single_Tuner = 1
    sdrplay_api_RspDuoMode_Dual_Tuner = 2
    sdrplay_api_RspDuoMode_Master = 4
    sdrplay_api_RspDuoMode_Slave = 8


class sdrplay_api_RspDuo_AmPortSelectT(IntEnum):
    sdrplay_api_RspDuo_AMPORT_1 = 1
    sdrplay_api_RspDuo_AMPORT_2 = 0


class sdrplay_api_RspDuoParamsT(ctypes.Structure):
    _fields_ = [
        ('extRefOutputEn', ctypes.c_int),
    ]


class sdrplay_api_RspDuo_ResetSlaveFlagsT(ctypes.Structure):
    _fields_ = [
        ('resetGainUpdate', ctypes.c_ubyte),
        ('resetRfUpdate', ctypes.c_ubyte),
    ]


class sdrplay_api_RspDuoTunerParamsT(ctypes.Structure):
    _fields_ = [
        ('biasTEnable', ctypes.c_ubyte),
        ('tuner1AmPortSel', ctypes.c_int),
        ('tuner1AmNotchEnable', ctypes.c_ubyte),
        ('rfNotchEnable', ctypes.c_ubyte),
        ('rfDabNotchEnable', ctypes.c_ubyte),
        ('resetSlaveFlags', sdrplay_api_RspDuo_ResetSlaveFlagsT),
    ]


class sdrplay_api_RspDx_AntennaSelectT(IntEnum):
    sdrplay_api_RspDx_ANTENNA_A = 0
    sdrplay_api_RspDx_ANTENNA_B = 1
    sdrplay_api_RspDx_ANTENNA_C = 2


class sdrplay_api_RspDx_HdrModeBwT(IntEnum):
    sdrplay_api_RspDx_HDRMODE_BW_0_200 = 0
    sdrplay_api_RspDx_HDRMODE_BW_0_500 = 1
    sdrplay_api_RspDx_HDRMODE_BW_1_200 = 2
    sdrplay_api_RspDx_HDRMODE_BW_1_700 = 3


class sdrplay_api_RspDxParamsT(ctypes.Structure):
    _fields_ = [
        ('hdrEnable', ctypes.c_ubyte),
        ('biasTEnable', ctypes.c_ubyte),
        ('antennaSel', ctypes.c_int),
        ('rfNotchEnable', ctypes.c_ubyte),
        ('rfDabNotchEnable', ctypes.c_ubyte),
    ]


class sdrplay_api_RspDxTunerParamsT(ctypes.Structure):
    _fields_ = [
        ('hdrBw', ctypes.c_int),
    ]


class sdrplay_api_RxChannelParamsT(ctypes.Structure):
    _fields_ = [
        ('tunerParams', sdrplay_api_TunerParamsT),
        ('ctrlParams', sdrplay_api_ControlParamsT),
        ('rsp1aTunerParams', sdrplay_api_Rsp1aTunerParamsT),
        ('rsp2TunerParams', sdrplay_api_Rsp2TunerParamsT),
        ('rspDuoTunerParams', sdrplay_api_RspDuoTunerParamsT),
        ('rspDxTunerParams', sdrplay_api_RspDxTunerParamsT),
    ]


class sdrplay_api_TransferModeT(IntEnum):
    sdrplay_api_ISOCH = 0
    sdrplay_api_BULK = 1


class sdrplay_api_FsFreqT(ctypes.Structure):
    _fields_ = [
        ('fsHz', ctypes.c_double),
        ('syncUpdate', ctypes.c_ubyte),
        ('reCal', ctypes.c_ubyte),
    ]


class sdrplay_api_SyncUpdateT(ctypes.Structure):
    _fields_ = [
        ('sampleNum', ctypes.c_uint),
        ('period', ctypes.c_uint),
    ]


class sdrplay_api_ResetFlagsT(ctypes.Structure):
    _fields_ = [
        ('resetGainUpdate', ctypes.c_ubyte),
        ('resetRfUpdate', ctypes.c_ubyte),
        ('resetFsUpdate', ctypes.c_ubyte),
    ]


class sdrplay_api_DevParamsT(ctypes.Structure):
    _fields_ = [
        ('ppm', ctypes.c_double),
        ('fsFreq', sdrplay_api_FsFreqT),
        ('syncUpdate', sdrplay_api_SyncUpdateT),
        ('resetFlags', sdrplay_api_ResetFlagsT),
        ('mode', ctypes.c_int),
        ('samplesPerPkt', ctypes.c_uint),
        ('rsp1aParams', sdrplay_api_Rsp1aParamsT),
        ('rsp2Params', sdrplay_api_Rsp2ParamsT),
        ('rspDuoParams', sdrplay_api_RspDuoParamsT),
        ('rspDxParams', sdrplay_api_RspDxParamsT),
    ]


class sdrplay_api_PowerOverloadCbEventIdT(IntEnum):
    sdrplay_api_Overload_Detected = 0
    sdrplay_api_Overload_Corrected = 1


class sdrplay_api_RspDuoModeCbEventIdT(IntEnum):
    sdrplay_api_MasterInitialised = 0
    sdrplay_api_SlaveAttached = 1
    sdrplay_api_SlaveDetached = 2
    sdrplay_api_SlaveInitialised = 3
    sdrplay_api_SlaveUninitialised = 4
    sdrplay_api_MasterDllDisappeared = 5
    sdrplay_api_SlaveDllDisappeared = 6


class sdrplay_api_EventT(IntEnum):
    sdrplay_api_GainChange = 0
    sdrplay_api_PowerOverloadChange = 1
    sdrplay_api_DeviceRemoved = 2
    sdrplay_api_RspDuoModeChange = 3
    sdrplay_api_DeviceFailure = 4


class sdrplay_api_GainCbParamT(ctypes.Structure):
    _fields_ = [
        ('gRdB', ctypes.c_uint),
        ('lnaGRdB', ctypes.c_uint),
        ('currGain', ctypes.c_double),
    ]


class sdrplay_api_PowerOverloadCbParamT(ctypes.Structure):
    _fields_ = [
        ('powerOverloadChangeType', ctypes.c_int),
    ]


class sdrplay_api_RspDuoModeCbParamT(ctypes.Structure):
    _fields_ = [
        ('modeChangeType', ctypes.c_int),
    ]


class sdrplay_api_EventParamsT(ctypes.Union):
    _fields_ = [
        ('gainParams', sdrplay_api_GainCbParamT),
        ('powerOverloadParams', sdrplay_api_PowerOverloadCbParamT),
        ('rspDuoModeParams', sdrplay_api_RspDuoModeCbParamT),
    ]


class sdrplay_api_StreamCbParamsT(ctypes.Structure):
    _fields_ = [
        ('firstSampleNum', ctypes.c_uint),
        ('grChanged', ctypes.c_int),
        ('rfChanged', ctypes.c_int),
        ('fsChanged', ctypes.c_int),
        ('numSamples', ctypes.c_uint),
    ]


sdrplay_api_StreamCallback_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_short),
    ctypes.POINTER(ctypes.c_short),
    ctypes.POINTER(sdrplay_api_StreamCbParamsT),
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
)


sdrplay_api_EventCallback_t = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(sdrplay_api_EventParamsT),
    ctypes.c_void_p,
)


class sdrplay_api_CallbackFnsT(ctypes.Structure):
    _fields_ = [
        ('StreamACbFn', sdrplay_api_StreamCallback_t),
        ('StreamBCbFn', sdrplay_api_StreamCallback_t),
        ('EventCbFn', sdrplay_api_EventCallback_t),
    ]


class sdrplay_api_ErrT(IntEnum):
    sdrplay_api_Success = 0
    sdrplay_api_Fail = 1
    sdrplay_api_InvalidParam = 2
    sdrplay_api_OutOfRange = 3
    sdrplay_api_GainUpdateError = 4
    sdrplay_api_RfUpdateError = 5
    sdrplay_api_FsUpdateError = 6
    sdrplay_api_HwError = 7
    sdrplay_api_AliasingError = 8
    sdrplay_api_AlreadyInitialised = 9
    sdrplay_api_NotInitialised = 10
    sdrplay_api_NotEnabled = 11
    sdrplay_api_HwVerError = 12
    sdrplay_api_OutOfMemError = 13
    sdrplay_api_ServiceNotResponding = 14
    sdrplay_api_StartPending = 15
    sdrplay_api_StopPending = 16
    sdrplay_api_InvalidMode = 17
    sdrplay_api_FailedVerification1 = 18
    sdrplay_api_FailedVerification2 = 19
    sdrplay_api_FailedVerification3 = 20
    sdrplay_api_FailedVerification4 = 21
    sdrplay_api_FailedVerification5 = 22
    sdrplay_api_FailedVerification6 = 23
    sdrplay_api_InvalidServiceVersion = 24


class sdrplay_api_ReasonForUpdateT(IntEnum):
    sdrplay_api_Update_None = 0
    sdrplay_api_Update_Dev_Fs = 1
    sdrplay_api_Update_Dev_Ppm = 2
    sdrplay_api_Update_Dev_SyncUpdate = 4
    sdrplay_api_Update_Dev_ResetFlags = 8
    sdrplay_api_Update_Rsp1a_BiasTControl = 16
    sdrplay_api_Update_Rsp1a_RfNotchControl = 32
    sdrplay_api_Update_Rsp1a_RfDabNotchControl = 64
    sdrplay_api_Update_Rsp2_BiasTControl = 128
    sdrplay_api_Update_Rsp2_AmPortSelect = 256
    sdrplay_api_Update_Rsp2_AntennaControl = 512
    sdrplay_api_Update_Rsp2_RfNotchControl = 1024
    sdrplay_api_Update_Rsp2_ExtRefControl = 2048
    sdrplay_api_Update_RspDuo_ExtRefControl = 4096
    sdrplay_api_Update_Master_Spare_1 = 8192
    sdrplay_api_Update_Master_Spare_2 = 16384
    sdrplay_api_Update_Tuner_Gr = 32768
    sdrplay_api_Update_Tuner_GrLimits = 65536
    sdrplay_api_Update_Tuner_Frf = 131072
    sdrplay_api_Update_Tuner_BwType = 262144
    sdrplay_api_Update_Tuner_IfType = 524288
    sdrplay_api_Update_Tuner_DcOffset = 1048576
    sdrplay_api_Update_Tuner_LoMode = 2097152
    sdrplay_api_Update_Ctrl_DCoffsetIQimbalance = 4194304
    sdrplay_api_Update_Ctrl_Decimation = 8388608
    sdrplay_api_Update_Ctrl_Agc = 16777216
    sdrplay_api_Update_Ctrl_AdsbMode = 33554432
    sdrplay_api_Update_Ctrl_OverloadMsgAck = 67108864
    sdrplay_api_Update_RspDuo_BiasTControl = 134217728
    sdrplay_api_Update_RspDuo_AmPortSelect = 268435456
    sdrplay_api_Update_RspDuo_Tuner1AmNotchControl = 536870912
    sdrplay_api_Update_RspDuo_RfNotchControl = 1073741824
    sdrplay_api_Update_RspDuo_RfDabNotchControl = 2147483648


class sdrplay_api_ReasonForUpdateExtension1T(IntEnum):
    sdrplay_api_Update_Ext1_None = 0
    sdrplay_api_Update_RspDx_HdrEnable = 1
    sdrplay_api_Update_RspDx_BiasTControl = 2
    sdrplay_api_Update_RspDx_AntennaControl = 4
    sdrplay_api_Update_RspDx_RfNotchControl = 8
    sdrplay_api_Update_RspDx_RfDabNotchControl = 16
    sdrplay_api_Update_RspDx_HdrBw = 32
    sdrplay_api_Update_RspDuo_ResetSlaveFlags = 64


class sdrplay_api_DbgLvl_t(IntEnum):
    sdrplay_api_DbgLvl_Disable = 0
    sdrplay_api_DbgLvl_Verbose = 1
    sdrplay_api_DbgLvl_Warning = 2
    sdrplay_api_DbgLvl_Error = 3
    sdrplay_api_DbgLvl_Message = 4


class sdrplay_api_DeviceT(ctypes.Structure):
    _fields_ = [
        ('SerNo', ctypes.c_char * SDRPLAY_MAX_SER_NO_LEN),
        ('hwVer', ctypes.c_ubyte),
        ('tuner', ctypes.c_int),
        ('rspDuoMode', ctypes.c_int),
        ('valid', ctypes.c_ubyte),
        ('rspDuoSampleFreq', ctypes.c_double),
        ('dev', ctypes.c_void_p),
    ]


class sdrplay_api_DeviceParamsT(ctypes.Structure):
    _fields_ = [
        ('devParams', ctypes.POINTER(sdrplay_api_DevParamsT)),
        ('rxChannelA', ctypes.POINTER(sdrplay_api_RxChannelParamsT)),
        ('rxChannelB', ctypes.POINTER(sdrplay_api_RxChannelParamsT)),
    ]


class sdrplay_api_ErrorInfoT(ctypes.Structure):
    _fields_ = [
        ('file', ctypes.c_char * 256),
        ('function', ctypes.c_char * 256),
        ('line', ctypes.c_int),
        ('message', ctypes.c_char * 1024),
    ]


sdrplay_api_Open_t = ctypes.CFUNCTYPE(ctypes.c_int)


sdrplay_api_Close_t = ctypes.CFUNCTYPE(ctypes.c_int)


sdrplay_api_ApiVersion_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float))


sdrplay_api_LockDeviceApi_t = ctypes.CFUNCTYPE(ctypes.c_int)


sdrplay_api_UnlockDeviceApi_t = ctypes.CFUNCTYPE(ctypes.c_int)


sdrplay_api_GetDevices_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(sdrplay_api_DeviceT),
    ctypes.POINTER(ctypes.c_uint),
    ctypes.c_uint,
)


sdrplay_api_SelectDevice_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(sdrplay_api_DeviceT))


sdrplay_api_ReleaseDevice_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(sdrplay_api_DeviceT),
)


sdrplay_api_GetErrorString_t = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_int)


sdrplay_api_GetLastError_t = ctypes.CFUNCTYPE(
    ctypes.POINTER(sdrplay_api_ErrorInfoT),
    ctypes.POINTER(sdrplay_api_DeviceT),
)


sdrplay_api_GetLastErrorByType_t = ctypes.CFUNCTYPE(
    ctypes.POINTER(sdrplay_api_ErrorInfoT),
    ctypes.POINTER(sdrplay_api_DeviceT),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_ulonglong),
)


sdrplay_api_DisableHeartbeat_t = ctypes.CFUNCTYPE(ctypes.c_int)


sdrplay_api_DebugEnable_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)


sdrplay_api_GetDeviceParams_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(sdrplay_api_DeviceParamsT)),
)


sdrplay_api_Init_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(sdrplay_api_CallbackFnsT),
    ctypes.c_void_p,
)


sdrplay_api_Uninit_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


sdrplay_api_Update_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


sdrplay_api_SwapRspDuoActiveTuner_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
)


sdrplay_api_SwapRspDuoDualTunerModeSampleRate_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
)


sdrplay_api_SwapRspDuoMode_t = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(sdrplay_api_DeviceT),
    ctypes.POINTER(ctypes.POINTER(sdrplay_api_DeviceParamsT)),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
)


# Each entry is (symbol, restype, argtypes).  The header declares every call as
# a function-pointer typedef whose name is the symbol plus "_t", so one typedef
# gives both the signature and what to look up in the library.
FUNCTIONS = [
    ('sdrplay_api_Open', ctypes.c_int, []),
    ('sdrplay_api_Close', ctypes.c_int, []),
    ('sdrplay_api_ApiVersion', ctypes.c_int, [ctypes.POINTER(ctypes.c_float)]),
    ('sdrplay_api_LockDeviceApi', ctypes.c_int, []),
    ('sdrplay_api_UnlockDeviceApi', ctypes.c_int, []),
    ('sdrplay_api_GetDevices', ctypes.c_int, [
        ctypes.POINTER(sdrplay_api_DeviceT),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_uint,
    ]),
    ('sdrplay_api_SelectDevice', ctypes.c_int, [ctypes.POINTER(sdrplay_api_DeviceT)]),
    ('sdrplay_api_ReleaseDevice', ctypes.c_int, [ctypes.POINTER(sdrplay_api_DeviceT)]),
    ('sdrplay_api_GetErrorString', ctypes.c_char_p, [ctypes.c_int]),
    ('sdrplay_api_GetLastError', ctypes.POINTER(sdrplay_api_ErrorInfoT), [
        ctypes.POINTER(sdrplay_api_DeviceT),
    ]),
    ('sdrplay_api_GetLastErrorByType', ctypes.POINTER(sdrplay_api_ErrorInfoT), [
        ctypes.POINTER(sdrplay_api_DeviceT),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulonglong),
    ]),
    ('sdrplay_api_DisableHeartbeat', ctypes.c_int, []),
    ('sdrplay_api_DebugEnable', ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    ('sdrplay_api_GetDeviceParams', ctypes.c_int, [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(sdrplay_api_DeviceParamsT)),
    ]),
    ('sdrplay_api_Init', ctypes.c_int, [
        ctypes.c_void_p,
        ctypes.POINTER(sdrplay_api_CallbackFnsT),
        ctypes.c_void_p,
    ]),
    ('sdrplay_api_Uninit', ctypes.c_int, [ctypes.c_void_p]),
    ('sdrplay_api_Update', ctypes.c_int, [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]),
    ('sdrplay_api_SwapRspDuoActiveTuner', ctypes.c_int, [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]),
    ('sdrplay_api_SwapRspDuoDualTunerModeSampleRate', ctypes.c_int, [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
    ]),
    ('sdrplay_api_SwapRspDuoMode', ctypes.c_int, [
        ctypes.POINTER(sdrplay_api_DeviceT),
        ctypes.POINTER(ctypes.POINTER(sdrplay_api_DeviceParamsT)),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]),
]
