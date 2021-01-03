import argparse

def combine_files(filenames, out_filename):
    with open(out_filename, 'w') as writer:
        with open(filenames[0]) as f:
            header = f.readline()
        writer.write(header)
        for filename in filenames:
            with open(filename) as f:
                header = f.readline()
                lines = f.readlines()
            writer.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/Users/jiangjunfeng/mainland/private/GMN_Chatbot/')
    args = parser.parse_args()

    filenames = [args.dir + 'data/%s_split_by_idname.csv' % year
                 for year in [2011, 2012, 2013, 2015, 2016, 2017, 2019]]
    out_filename = args.dir + 'data/from_2011_to_2019_split_by_idname.csv'
    combine_files(filenames, out_filename)