# PyMOL script to visualize CYP2C9 MD simulation candidate variants
# Load structure: fetch 1og5 or load your structure

# Color scheme
color gray80, all
color lightblue, resi 97-126  # SRS1
color lightgreen, resi 200-230  # SRS2
color lightorange, resi 290-320  # SRS4
color lightpink, resi 359-390  # SRS5
color salmon, resi 430-490  # SRS6

# Show heme
select heme, resn HEM
show sticks, heme
color red, heme

# Highlight Cys436 (heme-coordinating)
select cys436, resi 436
show spheres, cys436
color yellow, cys436

# Highlight variant positions
select var_98, resi 98
show spheres, var_98
color magenta, var_98
# p.G98V
select var_110, resi 110
show spheres, var_110
color magenta, var_110
# p.F110S
select var_124, resi 124
show spheres, var_124
color magenta, var_124
# p.R124W
select var_125, resi 125
show spheres, var_125
color magenta, var_125
# p.R125H
select var_132, resi 132
show spheres, var_132
color magenta, var_132
# p.R132Q
select var_132, resi 132
show spheres, var_132
color magenta, var_132
# p.R132W
select var_162, resi 162
show spheres, var_162
color magenta, var_162
# p.S162X
select var_299, resi 299
show spheres, var_299
color magenta, var_299
# p.T299R
select var_354, resi 354
show spheres, var_354
color magenta, var_354
# p.E354K
select var_359, resi 359
show spheres, var_359
color magenta, var_359
# p.I359L
select var_477, resi 477
show spheres, var_477
color magenta, var_477
# p.A477T
select var_489, resi 489
show spheres, var_489
color magenta, var_489
# p.P489S

# Labels
label resi 436 and name CA, "Cys436 (Heme)"

# Final view
zoom all
ray 1200, 900
png md_variants_structure.png
