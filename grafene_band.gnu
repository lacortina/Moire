set style data dots
set nokey
set xrange [0: 4.02418]
set yrange [-19.65283 : 17.03389]
set arrow from  1.70081, -19.65283 to  1.70081,  17.03389 nohead
set arrow from  2.55122, -19.65283 to  2.55122,  17.03389 nohead
set xtics ("G"  0.00000,"K"  1.70081,"M"  2.55122,"G"  4.02418)
 plot "grafene_band.dat"
