class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)
    def expand(self, border):
        self.w += 2*border
        self.h += 2*border
        self.x -= border
        self.y -= border
    def __contains__(self, other):
        return (
            other.x >= self.x and
            other.y >= self.y and
            other.x + other.w <= self.x + self.w and
            other.y + other.h <= self.y + self.h
            )
    def overlaps(r1, r2):
        if max(r1.x, r2.x) >= min(r1.x+r1.w, r2.x+r2.w): return False
        if max(r1.y, r2.y) >= min(r1.y+r1.h, r2.y+r2.h): return False
        return True
    def expandToIncludePoint(self, x, y):
        if x < self.x:
            self.w += self.x - x
            self.x = x
        if y < self.y:
            self.h += self.y - y
            self.y = y
        if x > self.x + self.w - 1:
            self.w += x - (self.x + self.w - 1)
        if y > self.y + self.h - 1:
            self.h += y - (self.y + self.h - 1)
    def expandToIncludeRect(self, other):
        self.expandToIncludePoint(other.x, other.y)
        self.expandToIncludePoint(other.x + other.w - 1, other.y + other.h - 1)
    def __repr__(self):
        return 'Rect(%d, %d, %d, %d)' % (self.x, self.y, self.w, self.h)

def iter_rect(x, y, w, h):
    assert w>=0 and h>=0
    for yy in xrange(y, y+h):
        for xx in xrange(x, x+w):
            yield (xx, yy)

if __name__ == '__main__':
    big = Rect(-3, 2, 180, 222)
    a = Rect(0, 10, 5, 15)
    b = Rect(2, 10, 1, 15)
    c = Rect(-1, 10, 1, 30)
    assert b in a
    assert a not in b
    assert a in big and b in big and c in big
    for r in [a, b, c]:
        assert r in big
        assert big.overlaps(r)
        assert r.overlaps(big)
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not c.overlaps(a)


    r1 = Rect( -40, -40, 5, 5 )
    r2 = Rect( -40-1, -40+5, 5, 500 )
    assert not r1.overlaps(r2)
    assert not r2.overlaps(r1)
    r1.y += 1;
    assert r1.overlaps(r2)
    assert r2.overlaps(r1)
    r1.x += 999;
    assert not r1.overlaps(r2)
    assert not r2.overlaps(r1)

    print 'Tests passed.'
