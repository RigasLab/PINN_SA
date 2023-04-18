import deepxde as dde

from deepxde import config
from deepxde.geometry.sampler import sample

import numpy as np

from GeometryBox.PeriodicHill.PHAuxFunctions import genH, genHArr, getBSLHill

class PHXDEGeometryBase(dde.geometry.geometry.Geometry):
    def __init__(
        self,
        ):

        xmin = [-4.5,0.0]
        xmax = [4.5,3.036]

        self.xmin = np.array(xmin, dtype=config.real(np))
        self.xmax = np.array(xmax, dtype=config.real(np))
        self.side_length = self.xmax - self.xmin

        #dim, bounding box, diameter
        super(PHXDEGeometryBase, self).__init__(
            2,(self.xmin, self.xmax), np.linalg.norm(self.side_length)
            )

        [xh,yh] = getBSLHill()
        xh -= 4.5

        diffX = xh[:-1] - xh[1:]
        diffY = yh[:-1] - yh[1:]

        d = diffX*diffX + diffY*diffY

        d = np.sqrt(d)

        cumul_perimeter = np.zeros(d.shape)
        cumul_perimeter[0] = d[0] 
        
        for i in range(1,len(d)):
            cumul_perimeter[i] = cumul_perimeter[i-1] + d[i]

        self.hill_perimeter = np.sum(d)
        self.cumul_hill_perimeter = cumul_perimeter
        self.xh = xh
        self.perimeter = self.hill_perimeter + 2*(self.xmax[1] - 1.0) + (self.xmax[0] - self.xmin[0])
        #Correct Later
        self.volume = np.prod(self.side_length)

    def inside(self, x):
        x = np.array(x, dtype=config.real(np))

        if (x.ndim == 1):
            bet_walls =  np.logical_and(
                np.greater_equal(x[1],genH(x[0]+4.5)),
                np.less_equal(x[1],self.xmax[1]),
            )
        else:
            bet_walls =  np.logical_and(
                np.greater_equal(x[:,1],genHArr(x[:,0]+4.5)),
                np.less_equal(x[:,1],self.xmax[1]),
            )

        in_bbd = np.logical_and(
            np.all(x >= self.xmin, axis=-1),
            np.all(x <= self.xmax, axis=-1),
        )

        # Between Walls and between bounding boxes 
        return np.logical_and(bet_walls, in_bbd)


    def on_boundary(self, x):
        x = np.array(x, dtype=config.real(np))

        if (x.ndim == 1):
            _on_boundary_tb = np.logical_or(
                np.isclose(x[1],self.xmax[1]),
                np.isclose(x[1],genH(x[0]+4.5)),
            )   

            _on_boundary_lr = np.logical_or(
                np.isclose(x[0],self.xmin[0]),
                np.isclose(x[0],self.xmax[0]),
            )
        else:
            _on_boundary_tb = np.logical_or(
                np.isclose(x[:,1],self.xmax[1]),
                np.isclose(x[:,1],genHArr(x[:,0]+4.5)),
            )   

            _on_boundary_lr = np.logical_or(
                np.isclose(x[:,0],self.xmin[0]),
                np.isclose(x[:,0],self.xmax[0]),
            )

        _on_boundary = np.logical_or(
            _on_boundary_tb,
            _on_boundary_lr,
        )

        return np.logical_and(self.inside(x), _on_boundary)


    # TO IMPLEMENT
    def boundary_normal(self, x):
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""
        raise NotImplementedError(
            "{}.boundary_normal to be implemented".format(self.idstr)
        )


    def periodic_point(self, x, component):
        y = np.copy(x)
        _on_xmin = np.isclose(y[:, component], self.xmin[component])
        _on_xmax = np.isclose(y[:, component], self.xmax[component])
        y[:, component][_on_xmin] = self.xmax[component]
        y[:, component][_on_xmax] = self.xmin[component]
        return y


    # TO IMPLEMENT
    def uniform_points(self, n, boundary=True):
        """Compute the equispaced point locations in the geometry."""
        print(
            "Warning: {}.uniform_points not implemented. Use random_points instead.".format(
                self.idstr
            )
        )
        return self.random_points(n)


    def random_points(self, n, random="pseudo"):
        #if (random == "reced"):
        #if (random == "Sobol"):
        #    x = np.empty((0, self.dim), dtype=config.real(np))
        #    vbbox = self.bbox[1] - self.bbox[0]

        #    a = 0.31
        #    x0 = 2
        #    b = -1/(x0+a)

        #    while len(x) < n:
        #        x_new = sample(n, self.dim, sampler="pseudo") * vbbox + self.bbox[0]
        #        dist = sample(n, 1, sampler="pseudo")
        #        t_b = sample(n, 1, sampler="pseudo")*2 - 1
        #        h = genHArr(x_new[:,0] + 4.5)
        #        y0 = (3.036 - h)/2

        #        ratY = 1 - (np.log(dist+a) + b*dist - np.log(a))
        #        ratY[:,0] *= y0

        #        IXt = (t_b > 0)
        #        IXb = (t_b <= 0)
                
        #        x_new[IXt[:,0],1] = 3.036 - ratY[IXt[:,0],0]
        #        x_new[IXb[:,0],1] = h[IXb[:,0]] + ratY[IXb[:,0],0]

        #        x = np.vstack((x, x_new[self.inside(x_new)]))
        #    return x[:n]
        #        
        #else:
            x = np.empty((0, self.dim), dtype=config.real(np))
            vbbox = self.bbox[1] - self.bbox[0]
            while len(x) < n:
                x_new = sample(n, self.dim, sampler=random) * vbbox + self.bbox[0]
                x = np.vstack((x, x_new[self.inside(x_new)]))
            return x[:n]


    #TO IMPLEMENT
    def uniform_boundary_points(self, n):
        """Compute the equispaced point locations on the boundary."""
        print(
            "Warning: {}.uniform_boundary_points not implemented. Use random_boundary_points instead.".format(
                self.idstr
            )
        )
        return self.random_boundary_points(n)

    def random_boundary_points(self, n, random="pseudo"):
        p = sample(n, 1, random)
        p *= self.perimeter

        vbbox = self.bbox[1] - self.bbox[0]
        x = sample(n, self.dim, sampler=random) * vbbox + self.bbox[0]

        for i,pt in enumerate(p):
            if (pt < self.hill_perimeter):
                dist = pt - self.cumul_hill_perimeter
                ix = np.argmin(np.abs(dist))

                x[i,0] = self.xh[ix+1]
                x[i,1] = genH(self.xh[ix+1]+4.5)

            elif ((pt >= self.hill_perimeter) and (pt < self.hill_perimeter + (self.xmax[1] - 1.0))):
                dist = pt - self.hill_perimeter

                x[i,0] = self.xmax[0]
                x[i,1] = 1.0 + dist
                  
            elif ((pt >= self.hill_perimeter + (self.xmax[1] - 1.0)) and (pt < self.perimeter - (self.xmax[1] - 1.0))):
                dist = pt - self.hill_perimeter - (self.xmax[1] - 1.0)

                x[i,0] = self.xmax[0] - dist
                x[i,1] = self.xmax[1]

            else:
                dist = self.perimeter - pt

                x[i,0] = self.xmin[0]
                x[i,1] = self.xmax[1] - dist

        return x


    #def on_boundary(self, x):
    #    _on_boundary = np.logical_or(
    #        np.any(np.isclose(x, self.xmin), axis=-1),
    #        np.any(np.isclose(x, self.xmax), axis=-1),
    #        )
    #    return np.logical_and(self.inside(x), _on_boundary)

    #def boundary_normal(self, x):
    #    _n = -np.isclose(x, self.xmin).astype(config.real(np)) + np.isclose(
    #        x, self.xmax
    #    )
    #    # For vertices, the normal is averaged for all directions
    #    idx = np.count_nonzero(_n, axis=-1) > 1
    #    if np.any(idx):
    #        print(
    #            f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
    #            "You may use PDE(..., exclusions=...) to exclude the vertices."
    #        )
    #        l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
    #        _n[idx] /= l
    #    return _n

    #def uniform_points(self, n, boundary=True):
    #    dx = (self.volume / n) ** (1 / self.dim)
    #    xi = []
    #    for i in range(self.dim):
    #        ni = int(np.ceil(self.side_length[i] / dx))
    #        if boundary:
    #            xi.append(
    #                np.linspace(
    #                    self.xmin[i], self.xmax[i], num=ni, dtype=config.real(np)
    #                )
    #            )
    #        else:
    #            xi.append(
    #                np.linspace(
    #                    self.xmin[i],
    #                    self.xmax[i],
    #                    num=ni + 1,
    #                    endpoint=False,
    #                    dtype=config.real(np),
    #                )[1:]
    #            )
    #    x = np.array(list(itertools.product(*xi)))
    #    if n != len(x):
    #        print(
    #            "Warning: {} points required, but {} points sampled.".format(n, len(x))
    #        )
    #    return x

    #def random_points(self, n, random="pseudo"):
    #    x = sample(n, self.dim, random)
    #    return (self.xmax - self.xmin) * x + self.xmin

    #def random_boundary_points(self, n, random="pseudo"):
    #    x = sample(n, self.dim, random)
    #    # Randomly pick a dimension
    #    rng = np.random.default_rng()
    #    rand_dim = rng.integers(self.dim, size=n)
    #    # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
    #    x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
    #    return (self.xmax - self.xmin) * x + self.xmin
