import pstats
import cProfile

import floatingse.floating as ff
import floatingse.semiInstance as se

# cProfile.run('ff.semiExample()','profout')
cProfile.run("se.psqp_optimal()", "profout")
p = pstats.Stats("profout")
n = 40
# Clean up filenames for the report
p.strip_dirs()

p.sort_stats("cumulative").print_stats(n)
p.sort_stats("time").print_stats(n)
