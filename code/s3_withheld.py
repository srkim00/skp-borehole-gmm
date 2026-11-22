# Event-withheld and station-withheld tests (Tables S5 and S6).
import pickle, time
import numpy as np, pandas as pd
from withheld_lib import *

SEED = 20261024


def folds(ids):
    p = np.random.default_rng(SEED).permutation(len(ids)); lab = np.empty(len(ids), int)
    for k, part in enumerate(np.array_split(p, 5)):
        lab[part] = k
    return dict(zip(ids, lab))


def needed_ims():
    s = []
    for im in EVAL:
        if im in SA:
            s += [x for x, _ in window(im)]
        else:
            s.append(im)
    return list(dict.fromkeys(s))


def run_test(kind, models):
    key = acc.event_id.values if kind == 'event' else acc.station_code.values
    fo = folds(np.unique(key[MAIN]))
    fold = np.array([fo.get(x, -1) for x in key])
    need = needed_ims()
    pred = {(m, im): np.full(len(acc), np.nan) for m in models for im in EVAL}
    for k in range(5):
        train = MAIN & (fold != k)
        coef = {im: fit_models(train, im, models) for im in need}
        for im in EVAL:
            te = test_population(im) & (fold == k)
            for m in models:
                if im in SA:
                    w = window(im); b = sum(wt * coef[x][m] for x, wt in w) / sum(wt for _, wt in w)
                else:
                    b = coef[im][m]
                pred[(m, im)][te] = XFULL[te][:, MODELS[m]] @ b
        print(kind, 'fold', k, 'done', flush=True)
    return pred


if __name__ == '__main__':
    t0 = time.time()
    pe = run_test('event', ('const', 'dist', 'rem'))
    ps = run_test('station', ('rem',))
    with open(f'{WORK}/withheld_predictions.pkl', 'wb') as fh:
        pickle.dump(dict(event=pe, station=ps), fh)
    rows, rows4, rowsb = [], [], []
    for im in EVAL:
        tp = test_population(im); ev = acc.event_id.values[tp]; y = res[im].values[tp]
        r = {'BSSA14': y}
        for m in ('const', 'dist', 'rem'):
            r[m + '_event'] = y - pe[(m, im)][tp]
        r['rem_station'] = y - ps[('rem', im)][tp]
        row = dict(IM=im, events=len(np.unique(ev)), records=int(tp.sum()))
        for k, v in r.items():
            row['RMSE_' + k] = event_rmse(v, ev)[0]; row['mean_' + k] = event_mean(v, ev)
        rows.append(row)
        mw = acc.Mw.values[tp] >= 4.0
        rows4.append(dict(IM=im, events=len(np.unique(ev[mw])), records=int(mw.sum()), RMSE_BSSA14=event_rmse(y[mw], ev[mw])[0],
                          RMSE_REM=event_rmse(r['rem_event'][mw], ev[mw])[0], mean_REM=event_mean(r['rem_event'][mw], ev[mw])))
    T5 = pd.DataFrame(rows); T6 = pd.DataFrame(rows4)
    T5.to_csv(f'{OUT}/withheld_rmse.csv', index=False); T6.to_csv(f'{OUT}/withheld_mw4.csv', index=False)
    pd.set_option('display.width', 200)
    print(T5[['IM', 'events', 'records', 'RMSE_BSSA14', 'RMSE_const_event', 'RMSE_dist_event', 'RMSE_rem_event', 'RMSE_rem_station']].round(3).to_string(index=False))
    print(T6.round(3).to_string(index=False))
    print('done', round(time.time() - t0), 's')
