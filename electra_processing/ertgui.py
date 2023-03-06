import os
import sys
import time
import re

from PySide6.QtGui import (
    QPalette,
    QColor,
    QScreen,
    QRegularExpressionValidator,
    QIcon,
    QPixmap,
    QIntValidator,
    QValidator,
)

from PySide6.QtCore import QSize, QTimer, QAbstractTableModel
from PySide6.QtSql import QSqlQueryModel, QSqlDatabase
from PySide6.QtCore import Qt, QAbstractTableModel

from ertprocess import process
from ertprocess import print_test


from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableView,
    QLabel,
    QFileDialog,
    QHeaderView,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
)


class NotEmptyAlphaValidator(QValidator):
    def validate(self, text, pos):
        if bool(text) and text.isalpha() and len(text) < 40:
            state = QIntValidator.Acceptable
        else:
            state = QIntValidator.Invalid
        return state, text, pos


class MainWindow(QMainWindow):
    def __init__(self, logfile_path):
        super().__init__()
        self.column_width = 240
        self.setWindowTitle("ELECTRA processing")

        pixmap = QPixmap('icon.png')
        pixmap = pixmap.scaled(self.column_width, 150, Qt.KeepAspectRatio)
        self.setWindowIcon(QIcon(pixmap))
        self.pixmap_widget = QLabel()
        self.pixmap_widget.setPixmap(pixmap)

        self.regex_positive_float = r"^[+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$"
        self.regex_positive_int = r"^[+]?[0-9]*?[0-9]+([eE][-+]?[0-9]+)?$"

        self.vs1 = QSpacerItem(0, 20, QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.vs2 = QSpacerItem(0, 80, QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.vs4 = QSpacerItem(0, 80, QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.hs0 = QSpacerItem(0, 0, QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.hs1 = QSpacerItem(40, 0, QSizePolicy.Ignored, QSizePolicy.Ignored)

        # browse button
        self.b_browse_files = QPushButton('Browse', default=False, autoDefault=True)
        self.b_browse_files.clicked.connect(self.browse_files)
        self.b_browse_files.setEnabled(True)
        self.b_browse_files.setStyleSheet(
            """QPushButton{
                    background-color: darkseagreen;
                    color: black;
            }"""
        )

        # browse button
        self.b_process_files = QPushButton('Process', default=False, autoDefault=True)
        self.b_process_files.clicked.connect(self.process_files)
        self.b_process_files.setEnabled(False)
        self.b_process_files.setStyleSheet(
            """QPushButton{
                    background-color: grey;
                    color: black;
            }"""
        )

        # log file
        self.logfile_label = QLabel('log file:\n' + logfile_path)
        self.logfile_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        # status
        self.status = QLineEdit()
        self.status.setText('status: ready')
        self.status.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.status.setEnabled(False)

        # output directory
        self.outdir_label = QLabel('Output Subfolder')
        self.outdir_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.outdir_edit.setText('processing')

        self.outdir_edit_nonEmptyValidator = NotEmptyAlphaValidator(self.outdir_edit)
        self.outdir_edit.setValidator(self.outdir_edit_nonEmptyValidator)

        self.outdir_layout = QVBoxLayout()
        self.outdir_layout.addWidget(self.outdir_label)
        self.outdir_layout.addWidget(self.outdir_edit)
        self.outdir_widget = QWidget()
        self.outdir_widget.setLayout(self.outdir_layout)

        # method
        self.method_label = QLabel('Method')
        self.method_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.method_M_m = QCheckBox()
        self.method_M_m.setText('M_m')
        self.method_FFT = QCheckBox()
        self.method_FFT.setText('FFT')
        self.method_FFT.setCheckState(Qt.Checked)
        self.method_FIT = QCheckBox()
        self.method_FIT.setText('FIT')

        self.method_layout = QVBoxLayout()
        self.method_layout.addWidget(self.method_label)
        self.method_layout.addWidget(self.method_M_m)
        self.method_layout.addWidget(self.method_FFT)
        self.method_layout.addWidget(self.method_FIT)
        self.method_widget = QWidget()
        self.method_widget.setLayout(self.method_layout)
        self.method_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.method_widget.setStyleSheet('background-color: #FFC3B8;')

        # export
        self.export_label = QLabel('Export')
        self.export_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.export_csv = QCheckBox()
        self.export_csv.setText('CSV')
        self.export_csv.setCheckState(Qt.Checked)
        self.export_simpeg = QCheckBox()
        self.export_simpeg.setText('Simpeg')
        self.export_res2dinv = QCheckBox()
        self.export_res2dinv.setText('Res2DInv')
        self.export_res2dinv.setCheckState(Qt.Checked)
        self.export_pygimli = QCheckBox()
        self.export_pygimli.setText('PyGimli')
        self.export_plot = QCheckBox()
        self.export_plot.setText('Plot')
        self.export_plot.setCheckState(Qt.Checked)

        self.export_layout = QVBoxLayout()
        self.export_layout.addWidget(self.export_label)
        self.export_layout.addWidget(self.export_csv)
        self.export_layout.addWidget(self.export_simpeg)
        self.export_layout.addWidget(self.export_res2dinv)
        self.export_layout.addWidget(self.export_pygimli)
        self.export_layout.addWidget(self.export_plot)
        self.export_widget = QWidget()
        self.export_widget.setLayout(self.export_layout)

        self.export_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.export_widget.setStyleSheet('background-color: #E2DAE7;')

        # export error
        self.err_export_label = QLabel('Export Err')
        self.err_export_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # base error
        self.err_base_check = QCheckBox()
        self.err_base_check.setText('Add base error')
        self.err_base_check.setCheckState(Qt.Checked)
        self.err_base_pct = QLineEdit()
        self.err_base_pct.setText('1')
        self.err_base_pct.setEnabled(True)
        self.err_base_pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.err_base_check.toggled.connect(self.err_base_pct.setEnabled)
        self.err_base_pct_validator = QRegularExpressionValidator(self.regex_positive_float, self.err_base_pct)
        self.err_base_pct.setValidator(self.err_base_pct_validator)
        self.err_base_layout = QHBoxLayout()
        self.err_base_layout.addWidget(self.err_base_check)
        self.err_base_layout.addWidget(self.err_base_pct)
        self.err_base_widget = QWidget()
        self.err_base_widget.setLayout(self.err_base_layout)
        # rec error
        self.err_rec_check = QCheckBox()
        self.err_rec_check.setText('Add rec error')
        self.err_rec_check.setCheckState(Qt.Checked)
        # self.err_rec_check.toggled.connect(self.rec_check.setEnabled)
        self.err_rec_layout = QHBoxLayout()
        self.err_rec_layout.addWidget(self.err_rec_check)
        self.err_rec_widget = QWidget()
        self.err_rec_widget.setLayout(self.err_rec_layout)
        # all error
        self.err_layout = QVBoxLayout()
        self.err_layout.addWidget(self.err_export_label)
        self.err_layout.addWidget(self.err_base_widget)
        self.err_layout.addWidget(self.err_rec_widget)
        self.err_widget = QWidget()
        self.err_widget.setLayout(self.err_layout)
        self.err_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.err_widget.setStyleSheet('background-color: #FBECFE;')

        # reciprocal
        self.rec_label = QLabel('Reciprocals')
        self.rec_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.rec_check = QCheckBox()
        self.rec_check.setText('Rec err %')
        self.rec_check.setCheckState(Qt.Checked)
        self.rec_max = QLineEdit()
        self.rec_max.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rec_max.setText('5')
        self.rec_max.setReadOnly(False)
        self.rec_validator = QRegularExpressionValidator(self.regex_positive_float, self.rec_max)
        self.rec_max.setValidator(self.rec_validator)
        self.rec_r = QCheckBox()
        self.rec_r.setText('r')
        self.rec_r.setCheckState(Qt.Checked)
        self.rec_rhoa = QCheckBox()
        self.rec_rhoa.setText('rhoa')
        self.rec_couple = QCheckBox()
        self.rec_couple.setText('Couple')
        self.rec_couple.setCheckState(Qt.Checked)
        self.rec_unpaired = QCheckBox()
        self.rec_unpaired.setText('Unpaired')
        self.rec_unpaired.setCheckState(Qt.Checked)
        self.rec_err_layout = QHBoxLayout()
        self.rec_err_layout.addWidget(self.rec_check)
        self.rec_err_layout.addWidget(self.rec_max)
        self.rec_err_widget = QWidget()
        self.rec_err_widget.setLayout(self.rec_err_layout)
        self.rec_quantities = QHBoxLayout()
        self.rec_quantities.addWidget(self.rec_r)
        self.rec_quantities.addWidget(self.rec_rhoa)
        self.rec_quantities_widget = QWidget()
        self.rec_quantities_widget.setLayout(self.rec_quantities)
        self.rec_layout = QVBoxLayout()
        self.rec_layout.addWidget(self.rec_label)
        self.rec_layout.addWidget(self.rec_quantities_widget)
        self.rec_layout.addWidget(self.rec_err_widget)
        self.rec_layout.addWidget(self.rec_couple)
        self.rec_layout.addWidget(self.rec_unpaired)
        self.rec_widget = QWidget()
        self.rec_widget.setLayout(self.rec_layout)
        self.rec_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.rec_widget.setStyleSheet('background-color: #E7FCC5;')

        self.rec_check.toggled.connect(self.rec_max.setEnabled)
        self.rec_check.toggled.connect(self.rec_r.setEnabled)
        self.rec_check.toggled.connect(self.rec_rhoa.setEnabled)
        self.rec_check.toggled.connect(self.rec_couple.setEnabled)
        self.rec_check.toggled.connect(self.rec_unpaired.setEnabled)

        # rhoa
        self.rhoa_check = QCheckBox()
        self.rhoa_check.setText('Rhoa range')
        self.rhoa_check.setCheckState(Qt.Checked)
        self.rhoa_min_label = QLabel('Min')
        self.rhoa_min_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.rhoa_min = QLineEdit()
        self.rhoa_min.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rhoa_min.setText('5')
        rhoa_min_validator = QRegularExpressionValidator(self.regex_positive_float, self.rec_max)
        self.rhoa_min.setValidator(rhoa_min_validator)
        self.rhoa_max_label = QLabel('Max')
        self.rhoa_max_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.rhoa_max = QLineEdit()
        self.rhoa_max.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.rhoa_max.setText('500')
        rhoa_max_validator = QRegularExpressionValidator(self.regex_positive_float, self.rec_max)
        self.rhoa_max.setValidator(rhoa_max_validator)
        self.rhoa_check.toggled.connect(self.rhoa_min.setEnabled)
        self.rhoa_check.toggled.connect(self.rhoa_max.setEnabled)
        self.rhoamin_layout = QHBoxLayout()
        self.rhoamin_layout.addWidget(self.rhoa_min_label)
        self.rhoamin_layout.addWidget(self.rhoa_min)
        self.rhoamin_widget = QWidget()
        self.rhoamin_widget.setLayout(self.rhoamin_layout)
        self.rhoamax_layout = QHBoxLayout()
        self.rhoamax_layout.addWidget(self.rhoa_max_label)
        self.rhoamax_layout.addWidget(self.rhoa_max)
        self.rhoamax_widget = QWidget()
        self.rhoamax_widget.setLayout(self.rhoamax_layout)
        self.rhoa_layout = QVBoxLayout()
        self.rhoa_layout.addWidget(self.rhoa_check)
        self.rhoa_layout.addWidget(self.rhoamax_widget)
        self.rhoa_layout.addWidget(self.rhoamin_widget)
        self.rhoa_widget = QWidget()
        self.rhoa_widget.setLayout(self.rhoa_layout)
        self.rhoa_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.rhoa_widget.setStyleSheet('background-color: #D6E3FA;')

        # ctc
        self.ctc_check = QCheckBox()
        self.ctc_check.setText('Max contact\nkohm')
        self.ctc_check.setCheckState(Qt.Checked)
        self.ctc_max = QLineEdit()
        self.ctc_max.setText('10')
        self.ctc_max.setEnabled(True)
        self.ctc_max.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ctc_check.toggled.connect(self.ctc_max.setEnabled)

        ctc_max_validator = QRegularExpressionValidator(self.regex_positive_float, self.ctc_max)
        self.ctc_max.setValidator(ctc_max_validator)

        self.ctc_layout = QHBoxLayout()
        self.ctc_layout.addWidget(self.ctc_check)
        self.ctc_layout.addWidget(self.ctc_max)
        self.ctc_widget = QWidget()
        self.ctc_widget.setLayout(self.ctc_layout)
        self.ctc_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.ctc_widget.setStyleSheet('background-color: #FDE8B4;')

        # v
        self.v_check = QCheckBox()
        self.v_check.setText('Min voltage\nmV')
        self.v_check.setCheckState(Qt.Unchecked)
        self.v_min = QLineEdit()
        self.v_min.setText('1')
        self.v_min.setEnabled(False)
        self.v_min.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.v_check.toggled.connect(self.v_min.setEnabled)

        self.v_layout = QHBoxLayout()
        self.v_layout.addWidget(self.v_check)
        self.v_layout.addWidget(self.v_min)
        self.v_widget = QWidget()
        self.v_widget.setLayout(self.v_layout)
        self.v_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.v_widget.setStyleSheet('background-color: #FDE8B4;')

        # stacking
        self.stk_check = QCheckBox()
        self.stk_check.setText('Stk err %')
        self.stk_check.setCheckState(Qt.Unchecked)
        self.stk_max = QLineEdit()
        self.stk_max.setText('5')
        self.stk_max.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.stk_max.setEnabled(False)
        self.stk_check.toggled.connect(self.stk_max.setEnabled)

        self.stk_layout = QHBoxLayout()
        self.stk_layout.addWidget(self.stk_check)
        self.stk_layout.addWidget(self.stk_max)
        self.stk_widget = QWidget()
        self.stk_widget.setLayout(self.stk_layout)
        self.stk_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.stk_widget.setStyleSheet('background-color: #FDE8B4;')

        # bad elec
        self.elecbad_check = QCheckBox()
        self.elecbad_check.setText('Bad elec')
        self.elecbad_check.setCheckState(Qt.Unchecked)
        self.elecbad = QLineEdit()
        self.elecbad.setText('')
        self.elecbad.setEnabled(False)
        self.elecbad.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.elecbad_check.toggled.connect(self.elecbad.setEnabled)

        self.elecbad_layout = QHBoxLayout()
        self.elecbad_layout.addWidget(self.elecbad_check)
        self.elecbad_layout.addWidget(self.elecbad)
        self.elecbad_widget = QWidget()
        self.elecbad_widget.setLayout(self.elecbad_layout)
        self.elecbad_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.elecbad_widget.setStyleSheet('background-color: #EBFEFD;')

        # shift elec
        self.shiftelec_check = QCheckBox()
        self.shiftelec_check.setText('Shift elec')
        self.shiftelec_check.setCheckState(Qt.Unchecked)
        self.shiftelec = QLineEdit()
        self.shiftelec.setText('0')
        self.shiftelec.setEnabled(False)
        self.shiftelec.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.shiftelec_check.toggled.connect(self.shiftelec.setEnabled)

        self.shiftelec_layout = QHBoxLayout()
        self.shiftelec_layout.addWidget(self.shiftelec_check)
        self.shiftelec_layout.addWidget(self.shiftelec)
        self.shiftelec_widget = QWidget()
        self.shiftelec_widget.setLayout(self.shiftelec_layout)
        self.shiftelec_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.shiftelec_widget.setStyleSheet('background-color: #EBFEFD;')

        # shift meas
        self.shiftmeas_check = QCheckBox()
        self.shiftmeas_check.setText('Shift meas')
        self.shiftmeas_check.setCheckState(Qt.Unchecked)
        self.shiftmeas = QLineEdit()
        self.shiftmeas.setText('0')
        self.shiftmeas.setEnabled(False)
        self.shiftmeas.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.shiftmeas_check.toggled.connect(self.shiftmeas.setEnabled)

        self.shiftmeas_layout = QHBoxLayout()
        self.shiftmeas_layout.addWidget(self.shiftmeas_check)
        self.shiftmeas_layout.addWidget(self.shiftmeas)
        self.shiftmeas_widget = QWidget()
        self.shiftmeas_widget.setLayout(self.shiftmeas_layout)
        self.shiftmeas_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.shiftmeas_widget.setStyleSheet('background-color: #EBFEFD;')

        # LEFT
        self.left = QVBoxLayout()
        self.left.addWidget(self.b_browse_files)
        self.left.addWidget(self.b_process_files)
        self.left.addWidget(self.outdir_widget)
        self.left.addItem(self.vs2)
        self.left.addWidget(self.method_widget)
        self.left.addWidget(self.export_widget)
        self.left.addWidget(self.logfile_label)
        self.left.addWidget(self.status)
        self.widgetv_left = QWidget()
        self.widgetv_left.setFixedWidth(240)
        self.widgetv_left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.widgetv_left.setLayout(self.left)

        # CENTER
        self.center = QVBoxLayout()
        self.center.addWidget(self.pixmap_widget)
        self.center.addWidget(self.err_widget)
        self.center.addWidget(self.rec_widget)
        self.widgetv_center = QWidget()
        self.widgetv_center.setFixedWidth(240)
        self.widgetv_center.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.widgetv_center.setLayout(self.center)

        # RIGHT
        self.right = QVBoxLayout()
        self.right.addWidget(self.rhoa_widget)
        self.right.addWidget(self.ctc_widget)
        self.right.addWidget(self.v_widget)
        self.right.addWidget(self.stk_widget)
        self.right.addWidget(self.elecbad_widget)
        self.right.addWidget(self.shiftelec_widget)
        self.right.addWidget(self.shiftmeas_widget)
        self.widgetv_right = QWidget()
        self.widgetv_right.setFixedWidth(240)
        self.widgetv_right.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.widgetv_right.setLayout(self.right)

        self.layouth = QHBoxLayout()
        self.layouth.addWidget(self.widgetv_left)
        self.layouth.addWidget(self.widgetv_center)
        self.layouth.addWidget(self.widgetv_right)

        self.widgeth = QWidget()
        self.widgeth.setLayout(self.layouth)
        self.setCentralWidget(self.widgeth)

    def browse_files(self):
        # disable so that the user had to wait during the processing

        dialog_files = QFileDialog()
        dialog_files.setFileMode(QFileDialog.ExistingFiles)
        filter = "ele (*.ele);;txt (*.txt);;txt2 (*.txt);;Data (*.Data);;dat (*.dat);;csv (*.csv)"
        selection = dialog_files.getOpenFileNames(self, "Open files", "C\\Desktop", filter)


        selected_files = selection[0]
        if len(selected_files) >= 1:
            selected_files = selected_files
            selected_type = selection[1].split(' ')[0]
        else:
            selected_files = None
            selected_type = None

        elecbad_text = self.elecbad.text()
        elecbad_text.replace(' ', ',').replace(';', ',').split(',')
        elecbad = [int(e) for e in elecbad_text]

        method = []
        if self.method_M_m.isChecked():
            method.append('M_m')
        if self.method_FFT.isChecked():
            method.append('fft')
        if self.method_FIT.isChecked():
            method.append('fit')
        if len(method) == 0:
            raise ValueError('must select at least one method')

        args = {
            'fname': selected_files,
            'ftype': selected_type,
            'outdir': self.outdir_edit.text(),
            'method': method,
            'rec_check': self.rec_check.isChecked(),
            'rec_max': float(self.rec_max.text()),
            'rec_r': self.rec_r.isChecked(),
            'rec_rhoa': self.rec_rhoa.isChecked(),
            'rec_couple': self.rec_couple.isChecked(),
            'rec_unpaired': self.rec_unpaired.isChecked(),
            'stk_check': self.stk_check.isChecked(),
            'stk_max': float(self.stk_max.text()),
            'rhoa_check': self.rhoa_check.isChecked(),
            'rhoa_min': float(self.rhoa_min.text()),
            'rhoa_max': float(self.rhoa_max.text()),
            'ctc_check': self.ctc_check.isChecked(),
            'ctc_max': float(self.ctc_max.text()) * 1000,
            'v_check': self.v_check.isChecked(),
            'v_min': float(self.v_min.text()),
            'elecbad_check': self.elecbad_check.isChecked(),
            'elecbad': elecbad,
            'err_base_check': self.err_base_check.isChecked(),
            'err_base_pct': float(self.err_base_pct.text()),
            'err_rec_check': self.err_rec_check.isChecked(),
            'shiftelec_check': self.shiftelec_check.isChecked(),
            'shiftelec': int(self.shiftelec.text()),
            'shiftmeas_check': self.shiftmeas_check.isChecked(),
            'shiftmeas': int(self.shiftmeas.text()),
            'export_csv': self.export_csv.isChecked(),
            'export_simpeg': self.export_simpeg.isChecked(),
            'export_pygimli': self.export_pygimli.isChecked(),
            'export_res2dinv': self.export_res2dinv.isChecked(),
            'plot': self.export_plot.isChecked(),
            # these are not implemented for electra
            'k_check': False,
            'k_max': 10000,
            'k_file': None,
            'err_stk_check': False,
        }

        args['rec_quantities'] = []
        if self.rec_r.isChecked():
            args['rec_quantities'].append('r')
        if self.rec_rhoa.isChecked():
            args['rec_quantities'].append('rhoa')

        args['w_err'] = any([args['err_base_check'], args['err_rec_check']])

        self.args = args
        print(self.args)

        if selected_files is not None and selected_type is not None:
            self.b_process_files.setStyleSheet(
                """QPushButton{
                        background-color: darkseagreen;
                        color: black;
                }"""
            )
            self.b_process_files.setEnabled(True)


    def process_files(self):
        self.b_process_files.setText('Processing ...')
        self.b_process_files.setEnabled(False)
        QApplication.processEvents()

        try:
            datasets = process(self.args)
            for ds in datasets:
                print(ds)

        except Exception as err:
            print('error')
            print(err)
            QApplication.processEvents()
            self.status.setText('status: error, see log')
            self.b_process_files.setText('ERR')
            self.b_process_files.setStyleSheet(
                """QPushButton{
                        background-color: lightcoral;
                        color: black;
                }"""
            )

        else:
            self.b_process_files.setText('OK')
            self.b_process_files.setStyleSheet(
                """QPushButton{
                        background-color: grey;
                        color: black;
                }"""
            )
        QApplication.processEvents()


if '__main__' == __name__:

    debug = False

    if debug:
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon('icon.png'))
        window = MainWindow(logfile_path='debug - see cmd')

        window.setStyleSheet("""
            QMainWindow{background-color: lightgray; border: 5px solid #0777ff; border-radius: 5px;}
            QHeaderView::section:horizontal {padding: 5; border: 2px solid}
        """)
        window.resize(800, 450)
        window.show()
        app.exec()

    else:

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        logfile_path = 'ElectraProcessing.log'
        with open(logfile_path, "w") as logfile:

            sys.stdout = logfile
            sys.stderr = logfile

            app = QApplication(sys.argv)
            app.setWindowIcon(QIcon('icon.png'))
            window = MainWindow(logfile_path=logfile_path)

            window.setStyleSheet("""
                QMainWindow{background-color: lightgray; border: 5px solid #0777ff; border-radius: 5px;}
                QHeaderView::section:horizontal {padding: 5; border: 2px solid}
            """)
            window.resize(800, 450)
            window.show()
            app.exec()

            sys.stdout = old_stdout
            sys.stderr = old_stderr
