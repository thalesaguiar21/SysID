from matplotlib.markers import MarkerStyle
from abc import ABC, abstractmethod
# import pdb


fig_layout = dict(left=0.05, bottom=0.125, right=0.98,
                  top=0.95, wspace=0.1, hspace=0.3)
XDATA = 0
YDATA = 1


class _Plotter(ABC):
    """ """

    def __init__(self, figure, axis, ylabel, xlabel):
        self.figure = figure
        self.axis = axis
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.measure_style = None
        self.estimation_style = None
        super().__init__()

    @abstractmethod
    def plot(self, estim, measure):
        pass

    def _configure_axis(self):
        axes = self.figure.gca()
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.grid(True)
        axes.legend()
        self.figure.subplots_adjust(**fig_layout)


class TimePlotter(_Plotter):
    """ """

    def __init__(self, fig, axis, ylabel='', yticks=None):
        super().__init__(fig, axis, ylabel, 'Tempo')
        self.yticks = yticks
        self.estimation_style = dict(linestyle='-',
                                     label='Residual', color='r')
        self.nonadap_style = dict(linestyle='-.',
                                  label='Estático', color='g')
        self.measure_style = dict(linestyle='--',
                                  label='Medição', color='c')

    def plot(self, nonadap, estim, measure):
        self.axis.plot(measure, **self.measure_style)
        self.axis.plot(estim, **self.nonadap_style)
        self.axis.plot(nonadap, **self.estimation_style)
        self._configure_axis()
        self._configure_ticks()
        return self.axis

    def _configure_ticks(self):
        axes = self.figure.gca()
        axes.set_yticks(self.yticks)


class MovePlotter(TimePlotter):

    def __init__(self, fig, axis, ylabel='', xlabel=''):
        super().__init__(fig, axis, ylabel, None)
        self.xlabel = xlabel
        self.markerstyle = MarkerStyle(marker='o', fillstyle='none')
        self.measure_style = dict(s=5, label='Medição', color='b')

    def plot(self, smooth, estim, measure):
        self.axis.scatter(_xdata(measure),
                          _ydata(measure),
                          marker=self.markerstyle,
                          **self.measure_style)
        self.axis.plot(_xdata(estim), _ydata(estim), **self.nonadap_style)
        self.axis.plot(_xdata(smooth), _ydata(smooth), **self.estimation_style)
        self._configure_axis()
        self._annotate_the_beggining(measure)

    def _annotate_the_beggining(self, measure):
        inicio = (measure[0, 0], measure[0, 1])
        pos_anotation = (inicio[0] - 2, inicio[1] - 1)
        self.axis.annotate('Início', xy=inicio, xytext=pos_anotation,
                           arrowprops=dict(facecolor='black', shrink=0.05))


def _xdata(data):
    return data[:, XDATA]


def _ydata(data):
    return data[:, YDATA]
