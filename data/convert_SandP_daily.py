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

# load S&P500 and Fama-French data
SandP500csv = open('SandP500_daily.csv', 'r')
SandP = []
for row in SandP500csv:
    SandP.append(row.strip().split(','))
SandP500 = SandP[1:] # strip header

FFcsv = open('FF6Portfolios.txt', 'r')
FF = []
for row in FFcsv:
    FF.append(row.strip().split())
FF_data = FF[3:] # strip header

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

# Get Fama-French dates 
FF_dates = [ff[0] for ff in FF_data]

# Get indices of daily S&P500 returns that appear in Fama-French data
indices = [idx for idx, s in zip(range(len(sp_dates)), sp_dates) if s in FF_dates]
FF_indices = [ff_idx for ff_idx, ff in zip(range(len(FF_dates)), FF_dates) if ff in sp_dates]
SandP500_matched = np.asarray(SandP500)[indices,:]
FF_data = np.array(FF_data)[FF_indices,:]

# Convert relevant assets to returns
print "Converting column", SandP[0][4], "From S&P 500."
SandP500_returns = prices_to_returns(SandP500_matched[:,4])

# write returns and dates to CSV files
print "Saving data"
FF_returns = [1,4,7,10,13,16]
np.savetxt("SandP500_returns.csv", SandP500_returns, delimiter=",")
np.savetxt("FF_returns.csv", np.array(FF_data)[1:,FF_returns].astype(float), delimiter=",")
np.savetxt("dates.csv", FF_data[1:,0].astype(int), delimiter=",")
1/0
print "Done."
