/*  HPGutil.c  ---  library of general-purpose utility functions	*/

/*
 Copyright (C) 2012 Henri P. Gavin
 
    HPGutil is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version. 

    HPGutil is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HPGutil.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "HPGutil.h"
#include <string.h>
#include <time.h>

#define DEBUG 0


/* 
 * COLOR - change color on the screen ... 
 * Screen   Color  Scheme  : 0 = white on black, 1 = bright
 * first digit= 3  for text color	  first digit= 4  for  background color
 * second digit codes:
 * 0=black, 1=red, 2=green, 3=gold, 4=blue, 5=magenta, 6=cyan, 7=white
 * http://en.wikipedia.org/wiki/ANSI_escape_code
 */
void color ( const int colorCode )	/*  change the screen color      */
{
#if ANSI_SYS
	fprintf (stderr, "\033[%02dm", colorCode );
	(void) fflush(stderr);
#endif
	return;
}


/*
 * TEXTCOLOR - change color of text and background
 * tColor : text color : one of 'k' 'r' 'g' 'y' 'b' 'm' 'c' 'w'
 * bColor : back color : one of 'k' 'r' 'g' 'y' 'b' 'm' 'c' 'w'
 * nbf    : 'n' = normal, 'b' = bright/bold, 'f' = faint
 * uline  : 'u' = underline
 * http://en.wikipedia.org/wiki/ANSI_escape_code
 */
void textColor ( const char tColor, const char bColor, const char nbf, const char uline )
{
#if ANSI_SYS
	fprintf (stderr, "\033[%02d",0);// Control Sequence Introducer & reset		
	// background colors
	if ( bColor == 'k' ) fprintf (stderr, ";%02d", 40 ); // black
	if ( bColor == 'r' ) fprintf (stderr, ";%02d", 41 ); // red
	if ( bColor == 'g' ) fprintf (stderr, ";%02d", 42 ); // green
	if ( bColor == 'y' ) fprintf (stderr, ";%02d", 43 ); // yellow
	if ( bColor == 'b' ) fprintf (stderr, ";%02d", 44 ); // blue
	if ( bColor == 'm' ) fprintf (stderr, ";%02d", 45 ); // magenta
	if ( bColor == 'c' ) fprintf (stderr, ";%02d", 46 ); // cyan
	if ( bColor == 'w' ) fprintf (stderr, ";%02d", 47 ); // white

	// text colors
	if ( tColor == 'k' ) fprintf (stderr, ";%02d", 30 ); // black
	if ( tColor == 'r' ) fprintf (stderr, ";%02d", 31 ); // red
	if ( tColor == 'g' ) fprintf (stderr, ";%02d", 32 ); // green
	if ( tColor == 'y' ) fprintf (stderr, ";%02d", 33 ); // yellow
	if ( tColor == 'b' ) fprintf (stderr, ";%02d", 34 ); // blue
	if ( tColor == 'm' ) fprintf (stderr, ";%02d", 35 ); // magenta
	if ( tColor == 'c' ) fprintf (stderr, ";%02d", 36 ); // cyan
	if ( tColor == 'w' ) fprintf (stderr, ";%02d", 37 ); // white

//	printf(" tColor = %c   bColor = %c   nbf = %c\n", tColor, bColor, nbf );
	if ( nbf    == 'b' ) fprintf (stderr, ";%02d",  1 ); // bright
	if ( nbf    == 'f' ) fprintf (stderr, ";%02d",  2 ); // faint

	if ( uline == 'u' )  fprintf (stderr, ";%02d", 4 );  // underline

	fprintf (stderr,"m");		// Select Graphic Rendition (SGR)

	(void) fflush(stderr);
#endif
	return;
}


/*
 * ERRORMSG -  write a diagnostic error message in color
 */
void errorMsg ( const char *errString )
{
	fprintf(stderr,"\n\n");
	fflush(stderr);
#if ANSI_SYS
	color(1); color(41); color(37);
#endif
	fprintf(stderr,"  %s  ", errString );
#if ANSI_SYS
	fflush(stderr);
	color(0);
#endif
	fprintf(stderr,"\n\n");
	return;
}


/* 
 * OPENFILE  -  open a file or print a diagnostic error message 
 */
FILE *openFile ( const char *path, const char *fileName, const char *mode, char *usage )
{
	FILE	*fp;
	char	pathToFile[MAXL], errMsg[MAXL];

	if (mode == 0)	return 0;

	sprintf(pathToFile,"%s%s", path, fileName );
#if DEBUG
	printf(" openFile ... file name = %s\n", pathToFile);
#endif
	if ((fp=fopen(pathToFile,mode)) == NULL ) { // open file 
		switch (*mode) {
		   sprintf(errMsg," openFile: ");
		   case 'w':
			sprintf(errMsg,"%s%s\n  usage: %s","cannot write to file: ", pathToFile, usage );
			break;
		   case 'r':
			sprintf(errMsg,"%s%s\n  usage: %s","cannot read from file: ", pathToFile, usage );
			break;
		   case 'a':
			sprintf(errMsg,"%s%s\n  usage: %s","cannot append to file: ", pathToFile, usage );
			break;
		   default:
			sprintf(errMsg,"%s%s\n  usage: %s","cannot open file: ", pathToFile, usage );
		}
		errorMsg ( errMsg );
		exit(1);
	} else {
#if DEBUG
	printf(" openFile ... fp = %x\n", fp);
#endif
	
		return fp;
	}
}


/* 
 * SCANLINE -  scan through a line until a 'a' is reached, like getline() 3feb94
 */
int scanLine ( FILE *fp, int lim, char *s, const char a ) 
{
       	int     c=0,  i=-1;

	while (--lim > 0 && (c=getc(fp)) != EOF && c != a)  s[++i] = c;
	s[++i]='\0';
	return i ;
}


/* 
 * SCANLABEL -  scan through a line until a '"' is reached, like getline()
 */
int scanLabel ( FILE *fp, int lim, char *s, const char a )
{
       	int     c=0,  i=-1;

	while (--lim > 0 && (c=getc(fp)) != EOF && c != a)
		;			// scan to first delimitter char
	while (--lim > 0 && (c=getc(fp)) != EOF && c != a) 
		s[++i] = c;		// read the label between delimitters
	s[++i]='\0';
	return i ;
}


/* 
 * SCANFILE -  count the number of lines of multi-column data in a data file,
 * skipping over "head_lines" lines of header information 
 */
int scanFile ( FILE *fp, int head_lines, int start_chnl, int stop_chnl )
{
	int	points = 0,
		i, chn, ok=1;	
	float	data_value;
	char	ch;
	
	// scan through the header
	for (i=1;i<=head_lines;i++)     while (( ch = getc(fp)) != '\n') ;

	// count the number of lines of data
	do {
		for ( chn=start_chnl; chn <= stop_chnl; chn++ ) {
			ok=fscanf(fp,"%f",&data_value);
			if (ok==1)      ++points;
		}
		if(ok>0) while (( ch = getc(fp)) != '\n') ; 
	} while (ok==1);

	points = (int) ( points / (stop_chnl - start_chnl + 1) );
	// printf ("%% %d data points\n", points);

	rewind (fp);

	return(points);
}


/* 
 * GETLINE -  get line form a stream into a character string, return length
 * from K&R	       3feb94
 */  
int getLine ( FILE *fp, int lim, char *s )
{
	int     c=0, i=0;

	while (--lim > 0 && (c=getc(fp)) != EOF && c != '\n' )
		s[i++] = c;
/*	if (c == '\n')  s[i++] = c;     */
	s[i++] = '\0';
	return(i);
}


/* 
 * getTime  parse a numeric time string similar to YYYYMMDDhhmmss 
 * The input variables y, m, d, hr, mn, sc are the indices of the string s[]
 * which start the YYYY, MM, DD, hh, mm, ss sections of the time string.  
 * The corresponding time is returned in "time_t" format.
 */  
time_t getTime( char s[], int y, int m, int d, int hr, int mn, int sc, int os )
{
        char   temp[16];

	struct tm t_tm;
	time_t  t_time;

	t_tm.tm_year = atoi( strncpy( temp, s+y, 4 ) )-1900;
	temp[2]='\0';
	t_tm.tm_mon  = atoi( strncpy( temp, s+m,  2 ) )-1;
	t_tm.tm_mday = atoi( strncpy( temp, s+d,  2 ) );
	t_tm.tm_hour = atoi( strncpy( temp, s+hr, 2 ) );
	t_tm.tm_min  = atoi( strncpy( temp, s+mn, 2 ) );
	t_tm.tm_sec  = atoi( strncpy( temp, s+sc, 2 ) )+os;

	/*  all times are Universal Time never daylight savings time */
	t_tm.tm_isdst = -1 ;

	t_time = mktime(&t_tm);      // normalize t_tm 

//	printf("%d ... %s", (int) t_time, ctime(&t_time) );

	return t_time;

}


/*
 * SHOW_PROGRESS  -   show the progress of long computations
 */
void showProgress ( int i, int n, int count )
{
	int	k,j, line_length = 55;
	float	percent_done;

	percent_done = (float)(i) / (float)(n);

	j = (int) ceil(percent_done*line_length);

	for (k=1;k<=line_length+13;k++)	fprintf(stderr,"\b");
	for (k=1;k<j;k++)		fprintf(stderr,">");
	for (k=j;k<line_length;k++)	fprintf(stderr," ");
	fprintf(stderr," %5.1f%%", percent_done*100.0 );
	fprintf(stderr," %5d", count );
	fflush(stderr);

	return;
}


/* 
 * SFERR  -  Display error message upon an erronous *scanf operation
 */
void sferr ( char s[] )
{
	char    errMsg[MAXL];

	sprintf(errMsg,">> Input Data file error while reading %s\n",s);
        errorMsg(errMsg);
	return;
}

#undef DEBUG
