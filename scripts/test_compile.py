import atlas

@atlas.generator
def nihao():
    a = Select([1, 2, 3])
    b = aha()
    return a

@atlas.generator
def aha():
    b = Select([1, 2,3])

for x in nihao.generate():
    print(x)


