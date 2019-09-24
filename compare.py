import argparse
from utilities.InterpolatedTimeSeries import InterpolatedTimeSeries
from utilities.plot import plot_any_field
from postprocess import get_step_and_info
from fenicstools import interpolate_nonmatching_mesh
import dolfin as df
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Average various files")
    parser.add_argument("-l", "--list", nargs="+", help="List of folders",
                        required=True)
    parser.add_argument("-f", "--fields", nargs="+", default=None,
                        help="Sought fields")
    parser.add_argument("-t", "--time", type=float, default=0, help="Time")
    parser.add_argument("--show", action="store_true", help="Show")
    args = parser.parse_args()

    tss = []
    for folder in args.list:
        ts = InterpolatedTimeSeries(folder, sought_fields=args.fields)
        tss.append(ts)
    Ntss = len(tss)

    all_fields_ = []
    for ts in tss:
        all_fields_.append(set(ts.fields))
    all_fields = list(set.intersection(*all_fields_))
    if args.fields is None:
        fields = all_fields
    else:
        fields = list(set.intersection(set(args.fields), set(all_fields)))

    f_in = []
    for ts in tss:
        f_in.append(ts.functions())

    # Using the first timeseries to define the spaces
    # Could be redone to e.g. a finer, structured mesh.
    # ref_mesh = tss[0].mesh
    ref_spaces = dict([(field, f.function_space()) for
                       field, f in f_in[0].items()])

    f = dict()
    for field, ref_space in ref_spaces.items():
        f[field] = df.Function(ref_space)
    
    for field in fields:
        for its, ts in enumerate(tss):
            if args.time is not None:
                step, time = get_step_and_info(ts, args.time)

            ts.update(f_in[its][field], field, step)
            f_loc = interpolate_nonmatching_mesh(f_in[its][field],
                                                 ref_spaces[field])
            f[field].vector()[:] += f_loc.vector().get_local()/Ntss

    for field in fields:
        df.plot(f[field])

    plt.show()


if __name__ == "__main__":
    main()
