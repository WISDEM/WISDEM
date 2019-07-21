# Peter Graf, 9/4/12

#  Cross platform xcel wrapper
# uses win32com on Windows, uses applescript on Mac

import os,sys,subprocess

def PlatformIsWindows():
    import platform
    if ("windows" in platform.platform().lower()):
        return True
    else:
        return False

if (PlatformIsWindows()):
    import win32com.client as win32

def ASrun(script):
    fullcmd = "osascript %s" % script
#    print "executing: ", fullcmd
    os.system(fullcmd)

def ASopenWorkbook(workbook):
    ln1 = "set p to \"%s\" " % workbook
#/Users/pgraf/work/wese/wese-6_13_12/wese2/examples/excel-mac/bosMod.xlsx"
    ln2 = "set a to POSIX file p"
    ln3  = "tell application \"Microsoft Excel\""
    ln4 = "open a"
    ln5 = "end tell"
    cmd = "%s\n  %s\n  %s\n  %s\n  %s\n"  % (ln1, ln2, ln3, ln4, ln5)
    f = file("xcelcmd.txt", "w")
    f.write(cmd)
    f.close()
    ASrun("xcelcmd.txt")

def AScloseExcel():
    ln1 = "tell application \"Microsoft Excel\""
    ln2 = "quit saving no"
    ln3 = "end tell"
    cmd = "%s\n  %s\n  %s\n"  % (ln1, ln2, ln3)
    f = file("xcelcmd.txt", "w")
    f.write(cmd)
    f.close()
    ASrun("xcelcmd.txt")


def ASTellXcel(cmdin):
# set value of cell "B1" to myvar & " test"
    ln0 = "set myvar to 0"
    ln1 = "tell application \"Microsoft Excel\""
    # cmdin goes here
    ln3 = "end tell"
    ln4 = "return myvar"
    cmd = " %s\n  %s\n  %s\n  %s\n  %s\n"  % (ln0, ln1, cmdin, ln3, ln4)
    f = file("xcelcmd.txt", "w")
    f.write(cmd)
    f.close()
    fullcmd = "osascript xcelcmd.txt"
#    print "exec ", fullcmd
    os.system("osascript xcelcmd.txt > myout")

def ASsetCell(irow,icol,value, worksheet):
    import string
    mycell = "%s%d" % (string.ascii_uppercase[icol-1], irow)
    mystr = "activate object worksheet \"%s\"  \n set value of cell \"%s\" to \"%s\"" % (worksheet, mycell, str(value))
    ASTellXcel(mystr)
    
def ASgetCell(irow, icol, worksheet):
    import string
    mycell = "%s%d" % (string.ascii_uppercase[icol-1], irow)
    mystr = "activate object worksheet \"%s\"  \n copy value of cell \"%s\" to myvar" % (worksheet, mycell)
    ASTellXcel(mystr)
    val = float(file("myout").readlines()[0].strip())
    return val

def AScountWorkbooks():
    cmd = "tell application \"Microsoft Excel\" \n set myvar to count each workbook \n return myvar \n end tell"
    os.system("osascript -e '%s'  > myout" % (cmd))
    val = int(file("myout").readlines()[0].strip())
    return val

def AScloseWorkbook():
    cmd = "tell application \"Microsoft Excel\" \n close active workbook saving no \n  end tell"
    os.system("osascript -e '%s'  > myout" % (cmd))

    
# error codes?
# 0: all is well
# 1: failed to open Excel
# 2: failed to open workbook
# 3: failed to close Excel

class ExcelWrapper (object):
    def __init__(self):
        self.windows = PlatformIsWindows()

    def openWorkbook(self, xlsfile):
        self.xlsfile = xlsfile
        try:
            if (self.windows):
                self.xl = win32.gencache.EnsureDispatch('Excel.Application')
        except:
            sys.stdout.write ("Error starting Excel\n")
            return 1
        try:
            if (self.windows):
                self.ss = self.xl.Workbooks.Open(self.xlsfile, ReadOnly=True)
            else:
                self.xl = ASopenWorkbook(self.xlsfile)

        except:
            sys.stdout.write ("Error opening Excel file '%s'\n" % self.xlsfile)        
            # If user doesn't have any other workbooks open in Excel, quit Excel
            if (self.windows):
                if (self.xl.Application.Workbooks.Count == 0):
                    self.xl.Application.Quit()
            return 2
        return 0

    def closeExcel(self):
        try:
            if (self.windows):
                self.ss.Close(False)
                # If user doesn't have any other workbooks open in Excel, quit Excel
                if (self.xl.Application.Workbooks.Count == 0):
                    self.xl.Application.Quit()
            else:
                AScloseExcel()
        except:
            print "failed to close Excel"
            return 3
        return 0

    def countWorkbooks(self):
        if (self.windows):
            return self.xl.Application.Workbooks.Count
        else:
            return AScountWorkbooks()

    def closeWorkbook(self):
        if (self.windows):
            self.ss.Close(False)
        else:
            AScloseWorkbook()
    
    def setCell(self,irow,icol,value, worksheet):
        """
        set an input cell in an Excel spreadsheet
        """
        if (self.windows):
            self.ss.Activate()
            self.xl.Worksheets(worksheet).Activate()
            sh = self.ss.ActiveSheet
            sh.Cells(irow,icol).Value = value
            sh.Calculate()
        else:
            ASsetCell(irow,icol,value, worksheet)
        return 0

    def getCell(self,irow,icol, worksheet):
        """
        get an output cell in a spreadsheet
        """
        if (self.windows):
            cval = self.ss.Worksheets(worksheet).Cells(irow,icol).Value
        else:
            cval = ASgetCell(irow, icol, worksheet)
        return float(cval)


#-----------------------------------------------------------

def example():

#    workbook = "Your path to an excel workbook"
    workbook = "/Users/pgraf/work/wese/wese-6_13_12/wese2/examples/excel_wrapper/blank.xlsx"    
    xcel = ExcelWrapper()
    xcel.openWorkbook(workbook)
    nbooks = xcel.countWorkbooks()
    print "%d workbooks open" % nbooks
    for n in range(4,10):
        xcel.setCell(1,1,n,"Sheet1")
        val, res = xcel.getCell(1,2,"Sheet1")
        print "n=%d val(1,2) = %d" % (n,val)
    xcel.closeWorkbook()
    xcel.closeExcel()


if __name__=="__main__":

    example()



    
