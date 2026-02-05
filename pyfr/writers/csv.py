

class CSVStream:
    def __init__(self, fname, *, header=None, nflush=100):
        # Append the '.csv' extension
        if not fname.endswith('.csv'):
            fname += '.csv'

        # Open file for appending
        self.outf = open(fname, 'a')

        # Output a header if required
        if self.outf.tell() == 0 and header:
            print(header, file=self.outf)

        self.nprint = 0
        self.nflush = nflush

    def __call__(self, *args):
        print(*args, sep=',', file=self.outf)

        # Check if flush needed
        self.nprint += 1
        if self.nprint % self.nflush == 0:
            self.outf.flush()
