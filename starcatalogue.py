"""Interface for retrieving data from a star catalogue."""

import sys
import time
import subprocess

def get_UCAC4(ra, ra_w, dec, dec_h, filename="ucac4.txt"):
    """Retrieve starmap data from UCAC4 catalog."""
    errorlog_fn = "starfield_errorlog%f.txt" % time.time()

    if sys.platform.startswith("win"):
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        # How to suppress crash notification dialog?, Jan 14,2004 -
        # Raymond Chen"s response [1]

        import ctypes
        SEM_NOGPFAULTERRORBOX = 0x0002  # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
        # subprocess_flags = 0x8000000 #win32con.CREATE_NO_WINDOW?
    # else:
        #subprocess_flags = 0
    # command="%s %f %f %f %f -h %s
    # %s"%(ucac_exe,self.ra_cent,self.dec_cent,self.ra_w,self.dec_w,ucac_data,ucac_out)

    command = "E:\\01_MasterThesis\\00_Code\\star_cats\\u4test.exe %f %f %f %f -h E:\\01_MasterThesis\\02_Data\\UCAC4 %s" % (
        ra, dec, ra_w, dec_h, filename)
    print(command)

    for _ in range(0, 5):
        retcode = subprocess.call(command)
        print("Retcode ", retcode)
        if retcode == 0:
            break
        with open(errorlog_fn, "at") as fout:
            fout.write("%f,\'%s\',%d\n" % (time.time(), command, retcode))

    with open(filename, "rt") as file:
        lines = file.readlines()
        print("Lines", len(lines))
    out = []
    for line in lines[1:]:

        ra_star = float(line[11:23])
        dec_star = float(line[23:36])
        mag_star = float(line[36:43])
        # print((r,d,m))
        out.append([ra_star, dec_star, mag_star])
    return out
