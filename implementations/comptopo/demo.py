#import sys
#sys.path.append('./dionysus/bindings/python./..')

import dionysus as d
s = d.Simplex([0, 1, 2])
print("Dimension:", s.dimension())
print(len(d.closure([s], 2)))