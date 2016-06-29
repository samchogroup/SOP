EXECUTABLE      := sop2.x
CUFILES_sm_20         := \
	energy.cu \
	GPUvars.cu \
	sop.cu

CU_DEPS         := \

CCFILES         := \
	io.cpp \
	param.cpp \
	random_generator.cpp \
	global.cpp \
	misc.cpp \

USECUDPP := 1

INCLUDES := -I/usr/local/cudpp/cudpp/include

###############################################################################
#
# Rules and targets

include common.mk

