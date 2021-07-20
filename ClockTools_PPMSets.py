

from numba import njit
import numpy as np

@njit
def clockScan(_channels,_timetags, clockChan, dataChan1, dataChan2, refChan):
    j = 0
    k = 0
    deriv = 8000 #increasing this gives less oscillations
    prop = .000000000000005
    #deriv = 1800
    #prop = .000000000005
    phi0 = 0
    phiold = 0
    filterr = 0
    clock0 = -1
    u = 0

    Clocks = np.zeros(len(_channels))
    dataTags = np.zeros(len(_channels))
    dataTagsR = np.zeros(len(_channels))

    ClockPortion = np.zeros(5000)
    for i in range(5000):
        if _channels[i] == clockChan:
            ClockPortion[j] = _timetags[i]
            j = j + 1
    j = 0

    ### Initial Estimates
    ClockPortion = ClockPortion[ClockPortion > 0]  #cut off extra zeros
    period = (ClockPortion[-1] - ClockPortion[0])/(len(ClockPortion) -1)
    freq = 1 / period



    RecoveredClocks = np.zeros(len(_channels))
    Periods = np.zeros(len(_channels))
    dualData = np.zeros((len(_channels),2))
    countM = np.zeros(len(_channels))
    for i in range(len(_channels)):
        if _channels[i] == clockChan:
            Clocks[j] = _timetags[i]
            currentClock = _timetags[i]
            # do clock recovery
            if clock0 == -1:
            #if j == 0:
                clock0 = currentClock - period
            phi0 = (currentClock - (clock0 + period)) / period
            filterr = phi0 + (phi0 - phiold) * deriv
            freq = freq - filterr * prop
            clock0 = clock0 + (1 / freq)  # add one period
            # clock1 = self.clock0 + self.period
            period = 1 / freq
            phiold = phi0
            RecoveredClocks[j] = clock0
            Periods[j] = period
            j = j + 1

        else: #data channels
            if _channels[i] == refChan:
                # do something with it later
                continue
            if clock0 ==-1:
                #no recovered clock available yet. Throw out that data
                continue
            else:
                tag = _timetags[i] - clock0
                tagR = _timetags[i] - currentClock
                dataTags[k] = tag
                dataTagsR[k] = tagR
                k = k + 1
                if _channels[i] == dataChan1:

                    dualData[u,0] = tag
                    countM[u] = _timetags[i] #this is for count rate analysis in another function
                if _channels[i] == dataChan2 and abs(tag - dualData[u,0]) < 3000: #2nd tag withing 1ns of 1st tag
                    dualData[u, 1] = tag
                    u = u + 1  # pair identified, get ready for next pair

    Clocks = Clocks[Clocks > 0]
    RecoveredClocks = RecoveredClocks[RecoveredClocks > 0]
    dataTags = dataTags[dataTags > 0]
    Periods = Periods[Periods > 0]
    dualData = dualData[dualData[:,0] > 0]
    countM = countM[countM > 0]

    #basis = np.linspace(Clocks[0],Clocks[-1],len(Clocks))
    #diffs = Clocks - basis
    return Clocks, RecoveredClocks, dataTags, dataTagsR, dualData, countM