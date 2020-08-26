/*   hpgUtils.h  ---  library of general-purpose utility functions       */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXL 512

#define ANSI_SYS	1	/*  compile for ANSI_SYS driver; 0: don't */
// ... requires ANSI.SYS and the line   DEVICE = C:\ANSI.SYS  in  C:\CONFIG.SYS


/* ---------------------------------------------------------------------------
COLOR - change color on the screen ... 
 Screen   Color  Scheme  : 0 = white on black, 1 = bright
 first digit= 3  for text color          first digit= 4  for  background color
 second digit codes:     1=red, 2=green, 3=gold, 4=blue, 5=purple, 6=lght blue
---------------------------------------------------------------------------- */
void color ( const int colorCode );	/*  change the screen color      */


/* ---------------------------------------------------------------------------
TEXTCOLOR - change color of text and background
 tColor : text color : one of 'k' 'r' 'g' 'y' 'b' 'm' 'c' 'w'
 bColor : back color : one of 'k' 'r' 'g' 'y' 'b' 'm' 'c' 'w'
 nbf    : 'n' = normal, 'b' = bright/bold, 'f' = faint
 uline  : 'u' = underline
 http://en.wikipedia.org/wiki/ANSI_escape_code
--------------------------------------------------------------------------- */
void textColor ( const char tColor, const char bColor, const char nbf, const char uline );


/* ---------------------------------------------------------------------------
ERRORMSG -  write a diagnostic error message in color
---------------------------------------------------------------------------- */
void errorMsg ( const char *errString );


/*  -------------------------------------------------------------------------
OPENFILE  -  open a file or print a diagnostic error message 
---------------------------------------------------------------------------- */
FILE *openFile (const char *path, const char *fileName, const char *mode, char *usage );


/* ---------------------------------------------------------------------------
SCANLINE -  scan through a line until a 'a' is reached, like getline() 3feb94
---------------------------------------------------------------------------- */
int scanLine ( FILE *fp, int lim, char *s, const char a );


/* ---------------------------------------------------------------------------
SCANLABEL -  scan through a line until a '"' is reached, like getline()
---------------------------------------------------------------------------- */
int scanLabel ( FILE *fp, int lim, char *s, const char a );

int scanFile ( FILE *fp, int head_lines, int start_chnl, int stop_chnl );


/*  ---------------------------------------------------------------------------
 * GETLINE -  get line form a stream into a character string, return length
 * from K&R               3feb94
 * ---------------------------------------------------------------------------
 */
int getLine ( FILE *fp, int lim, char *s );


/*  ---------------------------------------------------------------------------
 * getTime  parse a numeric time string of YYYYMMDDhhmmss 
 * The input variables y, m, d, hr, mn, sc are the indices of the string s[]
 * which start the YYYY, MM, DD, hh, mm, ss sections of the time string.  
 * An offset (os) in seconds is added to allow for correction between
 * time zones, UTC and GPS times.  
 * The corresponding time is returned in "time_t" format.
 * ---------------------------------------------------------------------------
 */
time_t getTime( char s[], int y, int m, int d, int hr, int mn, int sc, int os );


/*  ---------------------------------------------------------------------------
SHOW_PROGRESS  -   show the progress of long computations
--------------------------------------------------------------------------- */
void showProgress ( int i, int n, int count );


/*  ---------------------------------------------------------------------------
 * SFERR  -  Display error message upon an erronous *scanf operation
 * ------------------------------------------------------------------------- */
void sferr ( char s[] );
