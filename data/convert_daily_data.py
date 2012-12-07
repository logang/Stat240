import numpy as np

#----------------------------------------------------------------------
# The following script loads the S&P 500 and Fama-French data
# for the Stat240 final project, matches them by date, and
# calculates returns from the asset prices. 

# function to convert prices to returns
def prices_to_returns(p):
    np.asarray(p)
    p = p.astype(float)
    m = len(p)
    r = np.zeros((m-1,1))
    for i in xrange(m-1):
        r[i] = (p[i+1]-p[i])/p[i]
    return r

# function to load csv and txt data
def load_text(csvfile, sep=",", num_headerlines=0, header_return=None):
    loaded_csv = open(csvfile, 'r')
    out = []
    for row in loaded_csv:
        if sep is not None:
            out.append(row.strip().split(sep))
        else:
            out.append(row.strip().split())
    if header_return is not None:
        header = out[header_return]
    else:
        header = False
    out = out[num_headerlines:] # strip header
    return header, out

# Load S&P500, Fama-French and Libor data
SandP500_header, SandP500 = load_text('SandP500_daily.csv', sep=',',num_headerlines=1,header_return=0)
FF_header, FF_data = load_text('FF6Portfolios.txt', sep=None, num_headerlines=3,header_return=1)
Libor_header, Libor = load_text('LiborRates.csv', sep=",", num_headerlines=1,header_return=0)

# SandP500csv = open('SandP500_daily.csv', 'r')
# SandP = []
# for row in SandP500csv:
#     SandP.append(row.strip().split(','))
# SandP500 = SandP[1:] # strip header

# FFcsv = open('FF6Portfolios.txt', 'r')
# FF = []
# for row in FFcsv:
#     FF.append(row.strip().split())
# FF_data = FF[3:] # strip header


# Remove hyphens from S&P500 dates and reorder them
SandP_dates = [s[0] for s in SandP500]
sp_dates = []
for s in SandP_dates:
    s_all = s.split('-')
    whole = ''
    for part in s_all:
        whole += part
    sp_dates.append(whole)
sp_dates = sp_dates[::-1]

# Remove hyphens from Libor data and reorder them
Libor_dates = [l[0].split("\t")[0][1:11] for l in Libor]
Libor_returns = [l[0].split("\t")[1] for l in Libor]
libor_dates = []
for l in Libor_dates:
    l_all = l.split('-')
    whole = ''
    for part in l_all:
        whole += part
    libor_dates.append(whole)

# Get Fama-French dates 
FF_dates = [ff[0] for ff in FF_data]

# Get indices of daily S&P500 returns that appear in Fama-French data
indices = [idx for idx, s in zip(range(len(sp_dates)), sp_dates) if s in FF_dates]
FF_indices = [ff_idx for ff_idx, ff in zip(range(len(FF_dates)), FF_dates) if ff in sp_dates]
SandP500_matched = np.asarray(SandP500)[indices,:]
FF_data = np.array(FF_data)[FF_indices,:]

# Get indices of daily Libor returns that appear in Fama-French data
libor_indices = [idx for idx, l in zip(range(len(libor_dates)), libor_dates) if l in FF_dates]
Libor_matched = np.zeros(len(indices),)
Libor_matched_dates = np.array(libor_dates)[libor_indices]
matches = [item for item in Libor_matched_dates if item in FF_dates]
misses = [item for item in FF_dates if item not in matches]
FF_data = np.array(FF_data)[FF_indices,:]

# Find and replace missing dates with the previous date's data
j = 0
Libor_matched_dates = Libor_matched.copy()
for i in xrange(len(FF_dates)):
    if FF_dates[i] not in misses:
        Libor_matched[i] = Libor_returns[libor_indices[j]]
        Libor_matched_dates[i] = libor_dates[libor_indices[j]]
        j += 1
    else:
        Libor_matched[i] = Libor_returns[libor_indices[j]]
        Libor_matched_dates[i] = libor_dates[libor_indices[j]]

# Convert relevant assets to returns
print "Converting column", SandP500_header, "From S&P 500."
SandP500_returns = prices_to_returns(SandP500_matched[:,4])

# write returns and dates to CSV files
print "Saving data"
FF_returns = [1,4,7,10,13,16]
np.savetxt("SandP500_returns.csv", SandP500_returns, delimiter=",")
np.savetxt("Libor_returns.csv", Libor_matched, delimiter=",")
np.savetxt("FF_returns.csv", np.array(FF_data)[1:,FF_returns].astype(float), delimiter=",")
np.savetxt("dates.csv", FF_data[1:,0].astype(int), delimiter=",")

print "Done."
