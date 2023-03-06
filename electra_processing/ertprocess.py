"""
ERT PROCESSING
it gets arguments from cmd-line and/or dictionary (e.g., ert manager script)
it uses a ERT processing class that delegates to two dataframes for data and elec tables
it filters the data based on args; set very loose args values to avoid filtering.
"""

import os
import datetime
import argparse
import pandas as pd
import numpy as np
from ertio import read_electra_ele
from ertio import read_electra_custom_complete
from ertutils import output_file
from ertsinusoids import sinusoids_fft
from ertsinusoids import sinusoids_fit

import matplotlib.pyplot as plt

try:
    from numba import jit
except ImportError:
    numba_opt = False
else:
    numba_opt = True


def general_get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    filters = parse.add_argument_group('filters')
    electra = parse.add_argument_group('electra')
    adjustments = parse.add_argument_group('adjustments')
    err = parse.add_argument_group('output')
    outputs = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-fname', type=str, help='data file to process', nargs='+')
    main.add_argument('-ftype', type=str, help='data type to process', default='ele', choices=['ele', 'labrecque'])
    main.add_argument('-outdir', type=str, help='output directory', default='processing')

    # ELECTRA
    electra.add_argument(
        '-method', type=str, default='fft', choices=['M_m', 'fft', 'fit'],
        help='quantities for measurement value, mean if more than 1', nargs='+',
    )

    # FILTERS
    filters.add_argument('-v_check', action='store_true', default=False, help='activate v analysis')
    filters.add_argument('-v_min', type=float, default=1E-5, help='min voltage, V')

    filters.add_argument('-ctc_check', action='store_true', default=False, help='activate ctc analysis')
    filters.add_argument('-ctc_max', type=float, default=1E+4, help='max ctc, ohm')

    filters.add_argument('-stk_check', action='store_true', default=False, help='activate stk analysis')
    filters.add_argument('-stk_max', type=float, default=5, help='max stacking err, pct')

    filters.add_argument('-rec_check', action='store_true', default=True, help='activate reciprocal analysis')
    filters.add_argument('-rec_max', type=float, default=5, help='max reciprocal err, pct')
    filters.add_argument(
        '-rec_quantities', type=str, default='r', choices=['r', 'rhoa'],
        help='quantities for reciprocal err, mean if more than 1', nargs='+',
    )
    filters.add_argument('-rec_couple', action='store_true', default=True, help='couple reciprocals')
    filters.add_argument('-rec_unpaired', action='store_true', default=True, help='keep unpaired')

    filters.add_argument('-k_check', action='store_true', default=False, help='activate k analysis')
    filters.add_argument('-k_max', type=float, default=None, help='max geometrical factor, m')
    filters.add_argument('-k_file', type=str, help='file with geom factors and activates k and rhoa; self to keep; 2d or 3d to calc')

    filters.add_argument('-rhoa_check', action='store_true', default=True, help='activate rhoa analysis')
    filters.add_argument('-rhoa_min', default=0.5, type=float, help='rhoa min')
    filters.add_argument('-rhoa_max', default=10000, type=float, help='rhoa max)')

    filters.add_argument('-elecbad_check', action='store_true', default=True, help='activate elecbad analysis')
    filters.add_argument('-elecbad', type=int, help='electrodes to remove', nargs='+')

    # ERR
    err.add_argument('-err_base_check', action='store_true', help='if true, add this as a minimum/base error')
    err.add_argument('-err_base_pct', default=2, type=float, help='add this value as a base error')
    err.add_argument('-err_rec_check', action='store_true', help='if true, add reciprocal error')
    err.add_argument('-err_stk_check', action='store_true', help='if true, add stacking error')

    # OUTPUTS
    outputs.add_argument('-w_err', action='store_true', help='write the error (some combination of err base, rec, stk) to the inversion files')
    outputs.add_argument('-w_rhoa', action='store_true', help='if true, write rhoa')
    outputs.add_argument('-w_ip', action='store_true', help='if true, write phase')
    outputs.add_argument('-plot', action='store_true', help='plot data quality figures')
    outputs.add_argument('-cross', action='store_true', help='report on the cross validity of the filters')
    outputs.add_argument('-export_simpeg', action='store_true', help='if true, export in simpeg format')
    outputs.add_argument('-export_pygimli', action='store_true', help='if true, export in pygimli format')
    outputs.add_argument('-export_res2dinv', action='store_true', help='if true, export in res2dinv format')
    outputs.add_argument('-export_csv', action='store_true', help='if true, export a csv with the processing details')

    # ADJUSTMENTS
    adjustments.add_argument('-s_abmn', type=int, default=0, help='shift abmn')
    adjustments.add_argument('-s_meas', type=int, default=0, help='shift measurement number')
    adjustments.add_argument('-s_elec', type=int, default=0, help='shift electrode number')
    adjustments.add_argument('-f_elec', type=str, default=None, help='electrode coordinates: x y z')

    # GET ARGS
    args = parse.parse_args()
    print(args)

    # make suere a list is passed even when a single file is given
    if isinstance(args.fname, str):
        args['fname'] = [args.fname]


    return(args)


def process_rec(a: np.uint16, b: np.uint16, m: np.uint16, n: np.uint16, x: np.float64) -> (np.uint16, np.float64, np.float64, np.uint8):
    len_sequence = int(len(x))
    rec_num = np.zeros_like(x, dtype=np.uint16)
    rec_avg = np.zeros_like(x, dtype=np.float64)
    rec_err = np.zeros_like(x, dtype=np.float64)
    rec_fnd = np.zeros_like(x, dtype=np.uint8)
    for i in range(len_sequence):
        if rec_num[i] != 0:
            continue
        for j in range(i + 1, len_sequence):
            if (a[i] == m[j] and b[i] == n[j] and m[i] == a[j] and n[i] == b[j]):
                avg = (x[i] + x[j]) / 2
                err = abs(x[i] - x[j]) / abs(avg) * 100
                rec_num[i] = j + 1
                rec_num[j] = i + 1
                rec_avg[i] = avg
                rec_avg[j] = avg
                rec_err[i] = err
                rec_err[j] = err
                rec_fnd[i] = 1  # mark meas as direct
                rec_fnd[j] = 2  # mark meas as reciprocal (keep 0 for unpaired)
                break
    return(rec_num, rec_avg, rec_err, rec_fnd)


def numba_optimize_process_rec(process_rec):
    s = 'Tuple((uint16[:],float64[:],float64[:],uint8[:]))(uint16[:],uint16[:],uint16[:],uint16[:],float64[:])'
    process_rec = jit(
        signature_or_function=s,
        nopython=True,
        parallel=False,
        cache=True,
        fastmath=True,
        nogil=True,
    )(process_rec)
    return(process_rec)


def print_test(s):
    print(s)


def naive_pairs(l):
    for c in l:
        for r in l:
            if c == r:
                pass
            else:
                yield((c, r))


def crossvalidity(df, summary_col_header='valid'):
    """
    1 is valid and 0 is invalid, the table will report the number of rejections
    """
    if summary_col_header not in df.columns.tolist():
        df[summary_col_header] = df.all(axis=1)

    cols = df.columns.tolist()

    dfc = pd.DataFrame(index=cols, columns=cols)

    # off-diagonal: number of data points that both filters reject
    for (p0, p1) in naive_pairs(cols):
        common_rejections = np.sum((~df[[p0, p1]]).all(axis=1))
        dfc.loc[p0, p1] = common_rejections
        dfc.loc[p1, p0] = common_rejections

    # diagonal: number of data points that only that filter reject
    # remove the summary column, it would always match the other columns
    # ns: no summary
    cols.remove(summary_col_header)
    dfns = ~df[cols]
    sum_reject = dfns.sum(axis=1)
    dfns_unique = dfns[sum_reject == 1]
    colsns_unique_reject = dfns_unique.sum(axis=0)
    for c in cols:
        dfc.loc[c, c] = colsns_unique_reject[c]
    dfc.loc[summary_col_header, summary_col_header] = np.sum(~df[summary_col_header])
    return(dfc)


def process(args):

    print('processing')

    for f in args['fname']:
        print(f)

        if args['ftype'] == 'ele':

            ds = read_electra_custom_complete(f)

            sinusoids = ds.sinusoids.to_numpy()
            if 'fft' in args['method']:
                # fft
                amp_fft = sinusoids_fft(sinusoids, ds.meta['freq'], ds.meta['curr_dur'], ds.meta['sampling'])
                ds.data['fft_r'] = amp_fft / ds.data['curr']
                ds.data['fft_rhoa'] = ds.data['fft_r'] * ds.data['k']
                ds.data['fft_vs_picking_pcterr'] = abs(ds.data['fft_r'] - ds.data['r']) / ((ds.data['r'] + ds.data['fft_r']) / 2) * 100
            if 'fit' in args['method']:
                # fit
                amp_fit = sinusoids_fit(sinusoids, ds.meta['freq'], ds.meta['curr_dur'], ds.meta['sampling'])
                ds.data['fit_r'] = amp_fit / ds.data['curr']
                ds.data['fit_rhoa'] = ds.data['fit_r'] * ds.data['k']
                ds.data['fit_vs_picking_pcterr'] = abs(ds.data['fit_r'] - ds.data['r']) / ((ds.data['r'] + ds.data['fit_r']) / 2) * 100
            
            # redefined quantities based on the chosen methods;e
            chosen_r_columns = [mq + '_r' for mq in args['method']]
            ds.data['r'] = ds.data[chosen_r_columns].mean(axis=1).to_numpy()
            ds.data['rhoa'] = ds.data['r'] * ds.data['k']
            ds.data['v'] = ds.data['r'] * ds.data['curr']

        elif args['ftype'] == 'labrecque':
            ds = read_labreque(f)

        else:
            raise ValueError(selected_type, " have yet to be implemented")

        file_path, file_name = os.path.split(f)
        ds.meta['file_name'] = file_name
        ds.meta['file_path'] = file_path
        ds.meta['file_type'] = args['ftype']
        ds.meta['processing_datetime'] = datetime.datetime.now().isoformat(timespec='seconds')

        # k
        if args['k_file']:
            k_vec = pd.read_csv('k_file', index_col=None, header=None)
            ds.set_k(k_vec)
            # assuming rhoa may be wrong if k is set: recalculate it from k and r
            ds.data['rhoa'] = ds.data['k'] * ds.data['r']
        if args['k_check']:
            ds.data['k_valid'] = ds.data['k'] < float(args['k_max'])

        # rec
        if args['rec_check']:
            for q in args['rec_quantities']:
                print('checking reciprocal: ', q)
                a = ds.data['a'].to_numpy(dtype=np.uint16)
                b = ds.data['b'].to_numpy(dtype=np.uint16)
                m = ds.data['m'].to_numpy(dtype=np.uint16)
                n = ds.data['n'].to_numpy(dtype=np.uint16)
                x = ds.data[q].to_numpy(dtype=np.float64)
                x_avg = q + '_rec_avg'
                x_err = q + '_rec_err'
                rec_num, rec_avg, rec_err, rec_fnd = process_rec(a, b, m, n, x)
                ds.data['rec_num'] = rec_num
                ds.data['rec_fnd'] = rec_fnd
                ds.data[x_avg] = rec_avg
                ds.data[x_err] = rec_err
            rec_err_columns = [c for c in ds.data.columns if '_rec_err' in c]
            ds.data['rec_err'] = ds.data[rec_err_columns].mean(axis=1).to_numpy()
            ds.data['rec_valid'] = ds.data['rec_err'] < args['rec_max']

        # bad elecs
        if args['elecbad_check']:
            print('removing the data associated with these electrodes: ', args['elecbad'])
            ds.data['elec_valid'] = ~np.isin(ds.data[['a', 'b', 'm', 'n']], args['elecbad']).any(axis=1)

        # ctc
        if args['ctc_check']:
            ds.data['ctc_valid'] = ds.data['ctc'] < float(args['ctc_max'])

        # stk
        if args['stk_check']:
            ds.data['stk_valid'] = ds.data['stk'] < float(args['stk_max'])

        # rhoa
        if args['rhoa_check']:
            ds.data['rhoa_valid'] = ds.data['rhoa'].between(args['rhoa_min'], args['rhoa_max'])

        # combine filters
        filters = ['rec_valid', 'k_valid', 'rhoa_valid', 'ctc_valid', 'elec_valid']
        ds.data['valid'] = ds.data[filters].all(axis='columns')
        ds.default_types()

        # err estimate
        ds.data.insert(8, 'err', 0)
        if args['err_base_check']:
            ds.data['err'] += float(args['err_base_pct'])
        if args['err_rec_check']:
            ds.data['err'] += ds.data['rec_err']
        if args['err_stk_check']:
            ds.data['err'] += ds.data['stk']
        if args['w_err']:
            if any(ds.data['err'].isnull()):
                raise ValueError('err column in not valid for writing')
            if any(ds.data['err'] == 0):
                raise ValueError('err column contains 0s, which may hinder the inversion')

        # metadata of the analysis
        ds.meta['meas_out'] = ds.meta['meas_tot'] - sum(ds.data['valid'])
        ds.meta['meas_in'] = sum(ds.data['valid'])
        ds.meta['by_rec'] = ds.meta['meas_tot'] - sum(ds.data['rec_valid'])
        ds.meta['by_rhoa'] = ds.meta['meas_tot'] - sum(ds.data['rhoa_valid'])
        ds.meta['by_k'] = ds.meta['meas_tot'] - sum(ds.data['k_valid'])
        ds.meta['by_ctc'] = ds.meta['meas_tot'] - sum(ds.data['ctc_valid'])
        ds.meta['by_elecbad'] = ds.meta['meas_tot'] - sum(ds.data['elec_valid'])

        # rec coupling, should stay between error analysis and export
        # 1 is the flag for direct and 0 for unpaired
        if args['rec_couple']:
            directs = ds.data.loc[ds.data['rec_fnd'] == 1]
            if 'r_rec_avg' in directs.columns:
                directs['r_directs'] = directs['r']
                directs['r'] = directs['r_rec_avg']
            if 'rhoa_rec_avg' in directs.columns:
                directs['rhoa_directs'] = directs['rhoa']
                directs['rhoa'] = directs['rhoa_rec_avg']
        if (args['rec_couple'] and not args['rec_unpaired']):
            ds.data = directs
        elif args['rec_couple'] and args['rec_unpaired']:
            unpaireds = ds.data.loc[ds.data['rec_fnd'] == 0] 
            ds.data = pd.concat([directs, unpaireds])

        # export
        if args['export_csv']:
            fcsv = output_file(f, new_ext='.csv', directory=args['outdir'])
            with open(fcsv, 'w') as fopen:
                print(ds, file=fopen)
        if args['export_simpeg']:
            fubc = output_file(f, new_ext='.xyz', directory=args['outdir'])
            ds.to_ubc_xyz(fubc, meas_col='r', w_err=args['w_err'])
        if args['export_res2dinv']:
            finv = output_file(f, new_ext='.dat', directory=args['outdir'])
            ds.to_res2dinv(finv, meas_col='rhoa', w_err=args['w_err'])
        if args['export_pygimli']:
            finv = output_file(f, new_ext='_pygimli.dat', directory=args['outdir'])
            ds.to_bert(finv, w_err=args['w_err'], w_ip=args['w_ip'])

        if args['plot']:
            plot_columns = []
            # general
            if 'rhoa' in ds.data.columns:
                plot_columns.append('rhoa')
            if 'r' in ds.data.columns:
                plot_columns.append('r')
            if args['rec_check']:
                plot_columns.append('rec_err')
            if args['ctc_check']:
                plot_columns.append('ctc')
            if args['stk_check']:
                plot_columns.append('stk')
            if args['v_check']:
                plot_columns.append('v')
            if args['k_check']:
                plot_columns.append('k')

            # ELECTRA
            # plot avaialble methods
            if 'fft_rhoa' in ds.data.columns:
                plot_columns.append('fft_rhoa')
            elif 'fft_r' in ds.data.columns:
                plot_columns.append('fft_r')
            if 'fit_rhoa' in ds.data.columns:
                plot_columns.append('fit_rhoa')
            elif 'fit_r' in ds.data.columns:
                plot_columns.append('fit_r')
            if 'M_m_rhoa' in ds.data.columns:
                plot_columns.append('M_m_rhoa')
            elif 'M_m_r' in ds.data.columns:
                plot_columns.append('M_m_r')
            # compare methods
            if 'M_m_r' in ds.data.columns and 'fft_r' in ds.data.columns:
                ds.data['M_m_r_over_fft_r'] = ds.data['M_m_r'] / ds.data['fft_r']
                plot_columns.append('M_m_r_over_fft_r')
            if 'M_m_r' in ds.data.columns and 'fit_r' in ds.data.columns:
                ds.data['M_m_r_over_fit_r'] = ds.data['M_m_r'] / ds.data['fit_r']
                plot_columns.append('M_m_r_over_fit_r')
            if 'fft_r' in ds.data.columns and 'fit_r' in ds.data.columns:
                ds.data['fft_r_over_fit_r'] = ds.data['fft_r'] / ds.data['fit_r']
                plot_columns.append('fft_r_over_fit_r')

            # plot with flexible scale
            ds.plot_together(f, plot_columns, valid_column='valid', outdir=args['outdir'])

        # yield the ertds in case it is needed
        # or to confimir that the processing is done
        yield(ds)


if __name__ == '__main__':
    args = vars(general_get_cmd())
    print(args['fname'])
    for ds in process(args):
        print(ds)
