#ifndef MICROSTRAN_CONFIG_H
#define MICROSTRAN_CONFIG_H

#ifdef __WIN32__
# define MSTRANP_EXPORT __declspec(dllexport)
# define MSTRANP_IMPORT __declspec(dllimport)
#else
# ifdef HAVE_GCCVISIBILITY
#  define MSTRANP_EXPORT __attribute__ ((visibility("default")))
#  define MSTRANP_IMPORT
# else
#  define MSTRANP_EXPORT
#  define MSTRANP_IMPORT
# endif
#endif

#ifdef HAVE_GCCVISIBILITY
# define MSTRANP_LOCAL __attribute__ ((visibility("hidden")))
#else
# define MSTRANP_LOCAL
#endif

#ifdef MSTRANP_BUILD
# define MSTRANP_API extern MSTRANP_EXPORT
# define MSTRANP_DLL MSTRANP_EXPORT
#else
# define MSTRANP_API extern MSTRANP_IMPORT
# define MSTRANP_DLL MSTRANP_IMPORT
#endif

#if !defined(MSTRANP_API) || !defined(MSTRANP_EXPORT) || !defined(MSTRANP_IMPORT)
# error "NO MSTRANP_API, MSTRANP_EXPORT, MSTRANP_IMPORT DEFINED"
#endif

#ifdef WIN32
# define FRAME3DD_PATHSEP "\\"
# define FRAME3DD_DEFAULT_DATA_DIR "c:\\Program Files\\FRAME3DD"
#else
# define FRAME3DD_DEFAULT_DATA_DIR "/home/john/frame3dd/src/microstran"
# define FRAME3DD_PATHSEP "/"
#endif

#endif
