__all__ = ["CartesianFrame", "EllipseFrame"]


import numpy as np


class CartesianFrame:
    """A grid of X and Y values contained in a bbox
    """

    def __init__(self, bbox):
        # Store the new bounding box
        self._bbox = bbox
        # Get the range of x and y
        yi, xi = bbox.start[-2:]
        yf, xf = bbox.stop[-2:]
        height, width = bbox.shape[-2:]
        y = np.linspace(yi, yf - 1, height)
        x = np.linspace(xi, xf - 1, width)
        # Create the grid used to create the image of the frame
        self._X, self._Y = np.meshgrid(x, y)
        self._R = None
        self._R2 = None

    @property
    def shape(self):
        return self._bbox.shape

    @property
    def bbox(self):
        return self._bbox

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


class EllipseFrame(CartesianFrame):
    """Frame to scale the radius based on the parameters of an ellipse

    This frame is used to calculate the coordinates of the
    radius and radius**2 from a given center location,
    based on the semi-major axis, semi-minor axis, and rotation angle.
    It is also used to calculate the gradient wrt either the
    radius**2 or radius for all of the model parameters.
    """

    def __init__(self, y0, x0, major, minor, theta, bbox, r_min=1e-20):
        """Initialize the EllipseFrame

        Parameters
        ----------
        y0: `float`
            The y-center of the ellipse.
        x0: `float`
            The x-center of the ellipse.
        major: `float`
            The length of the semi-major axis.
        minor: `float`
            The length of the semi-minor axis.
        theta: `float`
            The counter-clockwise rotation angle
            from the semi-major axis.
        r_min: `float`
            The minimum value of the radius.
            This is used to prevent divergences that occur
            when calculating the gradient at radius == 0.
        """
        super().__init__(bbox)
        # Set some useful parameters for derivations
        sin = np.sin(theta)
        cos = np.cos(theta)

        # Rotate into the frame with xMajor as the x-axis
        # and xMinor as the y-axis
        self._xMajor = (self._X - x0)*cos + (self._Y - y0)*sin
        self._xMinor = -(self._X - x0)*sin + (self._Y - y0)*cos
        # The scaled major and minor axes
        self._xa = self._xMajor/major
        self._yb = self._xMinor/minor

        # Store parameters useful for gradient calculation
        self._y0, self._x0 = y0, x0
        self._theta = theta
        self._major = major
        self._minor = minor
        self._sin, self._cos = sin, cos
        self._bbox = bbox
        self._rMin = r_min
        # Store the scaled radius**2 and radius
        self._radius2 = self._xa**2 + self._yb**2
        self._radius = None

    def grad_x0(self, input_grad, use_r2):
        """The gradient of either the radius or radius**2 wrt. the x-center

        Parameters
        ----------
        input_grad: np.array
            Gradient of the likelihood wrt the component model
        use_r2: bool
            Whether to calculate the gradient of the radius**2 (``useR2==True``)
            or the radius (``useR2==False``).

        Returns
        -------
        result: float
            The gradient of the likelihood wrt x0.
        """
        grad = -self._xa*self._cos/self._major + self._yb*self._sin/self._minor
        if use_r2:
            grad *= 2
        else:
            r = self.R
            grad *= 1/r
        return np.sum(grad*input_grad)

    def grad_y0(self, input_grad, use_r2):
        """The gradient of either the radius or radius**2 wrt. the y-center

        Parameters
        ----------
        input_grad: np.array
            Gradient of the likelihood wrt the component model
        use_r2: bool
            Whether to calculate the gradient of the radius**2 (``useR2==True``)
            or the radius (``useR2==False``).

        Returns
        -------
        result: float
            The gradient of the likelihood wrt y0.
        """
        grad = -(self._xa*self._sin/self._major + self._yb*self._cos/self._minor)
        if use_r2:
            grad *= 2
        else:
            r = self.R
            grad *= 1/r
        return np.sum(grad*input_grad)

    def grad_major(self, input_grad, use_r2):
        """The gradient of either the radius or radius**2 wrt. the semi-major axis

        Parameters
        ----------
        input_grad: np.array
            Gradient of the likelihood wrt the component model
        use_r2: bool
            Whether to calculate the gradient of the radius**2 (``useR2==True``)
            or the radius (``useR2==False``).

        Returns
        -------
        result: float
            The gradient of the likelihood wrt the semi-major axis.
        """
        grad = - 2/self._major*self._xa**2
        if use_r2:
            grad *= 2
        else:
            r = self.R
            grad *= 1/r
        return np.sum(grad*input_grad)

    def grad_minor(self, input_grad, use_r2):
        """The gradient of either the radius or radius**2 wrt. the semi-minor axis

        Parameters
        ----------
        input_grad: np.array
            Gradient of the likelihood wrt the component model
        use_r2: bool
            Whether to calculate the gradient of the radius**2 (``useR2==True``)
            or the radius (``useR2==False``).

        Returns
        -------
        result: float
            The gradient of the likelihood wrt the semi-minor axis.
        """
        grad = -2/self._minor*self._yb**2
        if use_r2:
            grad *= 2
        else:
            r = self.R
            grad *= 1/r
        return np.sum(grad*input_grad)

    def grad_theta(self, input_grad, use_r2):
        """The gradient of either the radius or radius**2 wrt. the rotation angle

        Parameters
        ----------
        input_grad: np.array
            Gradient of the likelihood wrt the component model
        use_r2: bool
            Whether to calculate the gradient of the radius**2 (``useR2==True``)
            or the radius (``useR2==False``).

        Returns
        -------
        result: float
            The gradient of the likelihood wrt the rotation angle.
        """
        grad = self._xa*self._xMinor/self._major - self._yb*self._xMajor/self._minor
        if use_r2:
            grad *= 2
        else:
            r = self.R
            grad *= 1/r
        return np.sum(grad*input_grad)

    @property
    def x0(self):
        """The x-center"""
        return self._x0

    @property
    def y0(self):
        """The y-center"""
        return self._y0

    @property
    def major(self):
        """The semi-major axis"""
        return self._major

    @property
    def minor(self):
        """The semi-minor axis"""
        return self._minor

    @property
    def theta(self):
        """The rotation angle"""
        return self._theta

    @property
    def bbox(self):
        """The bounding box to old the model"""
        return self._bbox

    @property
    def R(self):
        """The radial coordinates of each pixel"""
        if self._radius is None:
            self._radius = np.sqrt(self.R2)
        return self._radius

    @property
    def R2(self):
        """The radius squared located at each pixel"""
        return self._radius2 + self._rMin**2
