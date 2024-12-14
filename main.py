import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QLabel, QMessageBox, QTextEdit, QSizePolicy,
    QGroupBox, QGridLayout, QLineEdit, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Arc
from matplotlib.figure import Figure

class ElectrostaticFieldApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.initial_xlim = None
        self.initial_ylim = None

    def init_ui(self):
        self.setWindowTitle('Визуализация электростатического поля и эквипотенциалов с диполем')
        self.setGeometry(100, 100, 1600, 900)

        instructions_label = QLabel('Введите параметры зарядов (x y q) в каждой строке:')
        instructions_label.setAlignment(Qt.AlignLeft)

        self.charges_input = QTextEdit()
        self.charges_input.setPlaceholderText("Пример:\n0 0 1\n1 0 -1")
        self.charges_input.setFixedHeight(150)

        grid_group = QGroupBox("Параметры сетки")
        grid_layout = QGridLayout()

        self.grid_min_input = QLineEdit("-10")
        self.grid_max_input = QLineEdit("10")
        self.grid_points_input = QLineEdit("200")

        grid_layout.addWidget(QLabel("Мин X и Y:"), 0, 0)
        grid_layout.addWidget(self.grid_min_input, 0, 1)
        grid_layout.addWidget(QLabel("Макс X и Y:"), 1, 0)
        grid_layout.addWidget(self.grid_max_input, 1, 1)
        grid_layout.addWidget(QLabel("Количество точек:"), 2, 0)
        grid_layout.addWidget(self.grid_points_input, 2, 1)

        grid_group.setLayout(grid_layout)

        potential_group = QGroupBox("Параметры эквипотенциалов")
        potential_layout = QGridLayout()

        self.show_potential_checkbox = QCheckBox("Отображать эквипотенциальные линии")
        self.show_potential_checkbox.setChecked(True)
        potential_layout.addWidget(self.show_potential_checkbox, 0, 0, 1, 2)

        potential_layout.addWidget(QLabel("Количество уровней:"), 1, 0)
        self.potential_levels_spinbox = QSpinBox()
        self.potential_levels_spinbox.setRange(1, 100)
        self.potential_levels_spinbox.setValue(20)
        potential_layout.addWidget(self.potential_levels_spinbox, 1, 1)

        potential_group.setLayout(potential_layout)

        dipole_group = QGroupBox("Параметры диполя")
        dipole_layout = QGridLayout()

        self.enable_dipole_checkbox = QCheckBox("Добавить диполь")
        self.enable_dipole_checkbox.setChecked(False)
        dipole_layout.addWidget(self.enable_dipole_checkbox, 0, 0, 1, 2)

        dipole_layout.addWidget(QLabel("Позиция X:"), 1, 0)
        self.dipole_x_input = QLineEdit("0")
        dipole_layout.addWidget(self.dipole_x_input, 1, 1)

        dipole_layout.addWidget(QLabel("Позиция Y:"), 2, 0)
        self.dipole_y_input = QLineEdit("0")
        dipole_layout.addWidget(self.dipole_y_input, 2, 1)

        dipole_layout.addWidget(QLabel("Модуль момента (p):"), 3, 0)
        self.dipole_p_input = QLineEdit("1")
        dipole_layout.addWidget(self.dipole_p_input, 3, 1)

        dipole_layout.addWidget(QLabel("Направление (°):"), 4, 0)
        self.dipole_angle_input = QLineEdit("0")
        dipole_layout.addWidget(self.dipole_angle_input, 4, 1)

        dipole_group.setLayout(dipole_layout)

        self.plot_button = QPushButton('Построить поле')
        self.plot_button.clicked.connect(self.plot_field)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion_notify)

        input_layout = QVBoxLayout()
        input_layout.addWidget(instructions_label)
        input_layout.addWidget(self.charges_input)
        input_layout.addWidget(grid_group)
        input_layout.addWidget(potential_group)
        input_layout.addWidget(dipole_group)
        input_layout.addWidget(self.plot_button)
        input_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout, 1)
        main_layout.addWidget(self.canvas, 3)

        self.setLayout(main_layout)
        self.show()

    def plot_field(self):
        try:
            charges_text = self.charges_input.toPlainText().strip()
            if not charges_text:
                raise ValueError("Необходимо ввести хотя бы один заряд.")

            charges = []
            for idx, line in enumerate(charges_text.split('\n'), start=1):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Строка {idx}: Ожидается три значения (x y q).")
                x_str, y_str, q_str = parts
                try:
                    x, y, q = float(x_str), float(y_str), float(q_str)
                except ValueError:
                    raise ValueError(f"Строка {idx}: x, y и q должны быть числами.")
                charges.append((x, y, q))

            if not charges:
                raise ValueError("Необходимо ввести хотя бы один заряд.")

            try:
                grid_min = float(self.grid_min_input.text())
                grid_max = float(self.grid_max_input.text())
                grid_points = int(self.grid_points_input.text())
                if grid_min >= grid_max:
                    raise ValueError("Мин должно быть меньше Макс.")
                if grid_points <= 0:
                    raise ValueError("Количество точек должно быть положительным.")
            except ValueError as ve:
                raise ValueError(f"Параметры сетки: {ve}")

            show_potential = self.show_potential_checkbox.isChecked()
            potential_levels = self.potential_levels_spinbox.value()

            dipole_enabled = self.enable_dipole_checkbox.isChecked()
            if dipole_enabled:
                try:
                    dipole_x = float(self.dipole_x_input.text())
                    dipole_y = float(self.dipole_y_input.text())
                    dipole_p = float(self.dipole_p_input.text())
                    dipole_angle_deg = float(self.dipole_angle_input.text())
                except ValueError:
                    raise ValueError("Параметры диполя должны быть числами.")

                dipole_angle_rad = np.deg2rad(dipole_angle_deg)
                dipole_px = dipole_p * np.cos(dipole_angle_rad)
                dipole_py = dipole_p * np.sin(dipole_angle_rad)
                dipole_p_vector = np.array([dipole_px, dipole_py])
            else:
                dipole_p_vector = np.array([0.0, 0.0])
                dipole_x = dipole_y = dipole_p = dipole_angle_deg = None

            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            ax = self.ax

            x = np.linspace(grid_min, grid_max, grid_points)
            y = np.linspace(grid_min, grid_max, grid_points)
            X, Y = np.meshgrid(x, y)
            Ex = np.zeros_like(X)
            Ey = np.zeros_like(Y)
            V = np.zeros_like(X)

            for charge in charges:
                x0, y0, q = charge
                dx = X - x0
                dy = Y - y0
                r_squared = dx**2 + dy**2
                r_squared[r_squared == 0] = 1e-20
                r = np.sqrt(r_squared)
                Ex += q * dx / (r_squared * r)
                Ey += q * dy / (r_squared * r)
                V += q / r

            if dipole_enabled:
                dx = X - dipole_x
                dy = Y - dipole_y
                r_squared = dx**2 + dy**2
                mask = r_squared < 1e-6
                r_squared[mask] = 1e-6
                r = np.sqrt(r_squared)
                rx = dx / r
                ry = dy / r
                p_dot_r = dipole_px * rx + dipole_py * ry
                Ex += (3 * p_dot_r * rx - dipole_px) / (r**3)
                Ey += (3 * p_dot_r * ry - dipole_py) / (r**3)
                V += (dipole_px * dx + dipole_py * dy) / (r**3)

            ax.streamplot(X, Y, Ex, Ey, color='k', density=1.5, linewidth=0.5, arrowsize=1)

            if show_potential:
                V_min, V_max = np.min(V), np.max(V)
                if V_min == V_max:
                    V_min -= 1
                    V_max += 1
                levels = np.linspace(V_min, V_max, potential_levels)
                potential_contours = ax.contour(X, Y, V, levels=levels, cmap='viridis', alpha=0.7)
                ax.clabel(potential_contours, inline=True, fontsize=8, fmt="%.2f")

            plotted_positive = False
            plotted_negative = False
            for charge in charges:
                x0, y0, q = charge
                if q > 0 and not plotted_positive:
                    ax.plot(x0, y0, 'ro', markersize=8, label='Положительный заряд')
                    plotted_positive = True
                elif q < 0 and not plotted_negative:
                    ax.plot(x0, y0, 'bo', markersize=8, label='Отрицательный заряд')
                    plotted_negative = True
                elif q > 0:
                    ax.plot(x0, y0, 'ro', markersize=8)
                elif q < 0:
                    ax.plot(x0, y0, 'bo', markersize=8)

            if dipole_enabled:
                ax.arrow(dipole_x, dipole_y,
                         dipole_px, dipole_py,
                         head_width=0.3, head_length=0.5, fc='g', ec='g',
                         length_includes_head=True, label='Диполь')

                E_at_dipole = np.array([0.0, 0.0])
                for charge in charges:
                    x0, y0, q = charge
                    dx_c = dipole_x - x0
                    dy_c = dipole_y - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_at_dipole += q * np.array([dx_c, dy_c]) / (r_sq * r)

                delta = 1e-5
                E_x_plus = 0.0
                E_x_minus = 0.0
                for charge in charges:
                    x0, y0, q = charge
                    dx_c = (dipole_x + delta) - x0
                    dy_c = dipole_y - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_x_plus += q * dx_c / (r_sq * r)

                    dx_c = (dipole_x - delta) - x0
                    dy_c = dipole_y - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_x_minus += q * dx_c / (r_sq * r)
                dE_x_dx = (E_x_plus - E_x_minus) / (2 * delta)

                E_x_plus = 0.0
                E_x_minus = 0.0
                for charge in charges:
                    x0, y0, q = charge
                    dx_c = dipole_x - x0
                    dy_c = (dipole_y + delta) - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_x_plus += q * dx_c / (r_sq * r)

                    dx_c = dipole_x - x0
                    dy_c = (dipole_y - delta) - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_x_minus += q * dx_c / (r_sq * r)
                dE_x_dy = (E_x_plus - E_x_minus) / (2 * delta)

                E_y_plus = 0.0
                E_y_minus = 0.0
                for charge in charges:
                    x0, y0, q = charge
                    dx_c = (dipole_x + delta) - x0
                    dy_c = dipole_y - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_y_plus += q * dy_c / (r_sq * r)

                    dx_c = (dipole_x - delta) - x0
                    dy_c = dipole_y - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_y_minus += q * dy_c / (r_sq * r)
                dE_y_dx = (E_y_plus - E_y_minus) / (2 * delta)

                E_y_plus = 0.0
                E_y_minus = 0.0
                for charge in charges:
                    x0, y0, q = charge
                    dx_c = dipole_x - x0
                    dy_c = (dipole_y + delta) - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_y_plus += q * dy_c / (r_sq * r)

                    dx_c = dipole_x - x0
                    dy_c = (dipole_y - delta) - y0
                    r_sq = dx_c**2 + dy_c**2
                    if r_sq < 1e-6:
                        continue
                    r = np.sqrt(r_sq)
                    E_y_minus += q * dy_c / (r_sq * r)
                dE_y_dy = (E_y_plus - E_y_minus) / (2 * delta)

                grad_E = np.array([[dE_x_dx, dE_x_dy],
                                   [dE_y_dx, dE_y_dy]])

                F = dipole_p_vector @ grad_E
                tau = dipole_p_vector[0] * E_at_dipole[1] - dipole_p_vector[1] * E_at_dipole[0]

                force_scale = 0.5
                ax.arrow(dipole_x, dipole_y,
                         force_scale * F[0], force_scale * F[1],
                         head_width=0.3, head_length=0.5, fc='m', ec='m',
                         length_includes_head=True, label='Сила на диполь')

                if tau != 0:
                    if tau > 0:
                        theta1, theta2 = 0, 180
                        direction = 1
                    else:
                        theta1, theta2 = 180, 360
                        direction = -1
                    arc = Arc((dipole_x, dipole_y), 1, 1, angle=0, theta1=theta1, theta2=theta2,
                              edgecolor='c', linewidth=2)
                    ax.add_patch(arc)
                    arrow_angle = (theta1 + theta2) / 2
                    arrow_x = dipole_x + 0.5 * np.cos(np.deg2rad(arrow_angle))
                    arrow_y = dipole_y + 0.5 * np.sin(np.deg2rad(arrow_angle))
                    dx_arrow = 0.2 * np.cos(np.deg2rad(arrow_angle + 90 * direction))
                    dy_arrow = 0.2 * np.sin(np.deg2rad(arrow_angle + 90 * direction))
                    ax.arrow(arrow_x, arrow_y, dx_arrow, dy_arrow,
                             head_width=0.1, head_length=0.2, fc='c', ec='c',
                             length_includes_head=True)

                if dipole_enabled:
                    ax.text(dipole_x, dipole_y + 1, f'F = ({F[0]:.2f}, {F[1]:.2f})', color='m')
                    ax.text(dipole_x, dipole_y - 1.5, f'τ = {tau:.2f}', color='c')

            ax.set_xlim(grid_min, grid_max)
            ax.set_ylim(grid_min, grid_max)
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Электростатическое поле, эквипотенциалы и диполь')
            ax.grid(True)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc='upper right')

            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def on_button_press(self, event):
        if event.button == 3 and event.inaxes:
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
            self.initial_xlim = self.ax.get_xlim()
            self.initial_ylim = self.ax.get_ylim()

    def on_button_release(self, event):
        if event.button == 3:
            self.is_panning = False

    def on_motion_notify(self, event):
        if self.is_panning and event.inaxes:
            dx = self.pan_start_x - event.xdata
            dy = self.pan_start_y - event.ydata
            new_xlim = [self.initial_xlim[0] + dx, self.initial_xlim[1] + dx]
            new_ylim = [self.initial_ylim[0] + dy, self.initial_ylim[1] + dy]
            grid_min = float(self.grid_min_input.text())
            grid_max = float(self.grid_max_input.text())
            x_range = new_xlim[1] - new_xlim[0]
            y_range = new_ylim[1] - new_ylim[0]

            if new_xlim[0] < grid_min:
                new_xlim = [grid_min, grid_min + x_range]
            if new_xlim[1] > grid_max:
                new_xlim = [grid_max - x_range, grid_max]
            if new_ylim[0] < grid_min:
                new_ylim = [grid_min, grid_min + y_range]
            if new_ylim[1] > grid_max:
                new_ylim = [grid_max - y_range, grid_max]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw()

    def on_scroll(self, event):
        if not hasattr(self, 'ax'):
            return

        ax = self.ax
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        try:
            grid_min = float(self.grid_min_input.text())
            grid_max = float(self.grid_max_input.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Параметры сетки должны быть числами.")
            return

        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        x_range = (x_max - x_min) / 2
        y_range = (y_max - y_min) / 2

        scale_factor = 1.2 if event.button == 'up' else 0.8

        new_x_range = x_range * scale_factor
        new_y_range = y_range * scale_factor

        new_x_min = max(grid_min, x_center - new_x_range)
        new_x_max = min(grid_max, x_center + new_x_range)
        new_y_min = max(grid_min, y_center - new_y_range)
        new_y_max = min(grid_max, y_center + new_y_range)

        min_range = 0.1
        if (new_x_max - new_x_min) < min_range or (new_y_max - new_y_min) < min_range:
            return

        ax.set_xlim([new_x_min, new_x_max])
        ax.set_ylim([new_y_min, new_y_max])

        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    ex = ElectrostaticFieldApp()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
