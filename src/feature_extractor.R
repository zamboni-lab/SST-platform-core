
library(xcms)
library(MassSpecWavelet)

path_to_files = "/Users/andreidm/ETH/projects/feature_extractor/data/CsI_NaI_best_conc_mzXML/"

raw_data_files = list.files(path_to_files, recursive = TRUE, full.names=TRUE)

raw_data = readMSData(files = raw_data_files, msLevel.=1)

rd_list = as.list(raw_data@assayData)

rd_list$F1

# try MassSpecWavelet 
scales <- seq(1, 64, 2)

peak_number = 5

int_data = rd_list[[peak_number]]@intensity
mz_data = rd_list[[peak_number]]@mz

plot(mz_data,int_data,type="l")


wCoefs <- cwt(data, scales = scales, wavelet = "mexh")

wCoefs <- cbind(as.vector(data), wCoefs)

xTickInterval <- 1000
plotRange <- c(1, length(data))
image(plotRange[1]:plotRange[2], scales, wCoefs[plotRange[1]:plotRange[2],], col=terrain.colors(256), axes=FALSE, xlab='m/z index', ylab='CWT coefficient scale', main='CWT coefficients')
axis(1, at=seq(plotRange[1], plotRange[2], by=xTickInterval))
axis(2, at=c(1, seq(10, 64, by=10)))
box()

colnames(wCoefs) <- c(0, scales)
localMax <- getLocalMaximumCWT(wCoefs)

plotLocalMax(localMax, wCoefs, range=newPlotRange)

ridgeList <- getRidge(localMax)

plotRidgeList(ridgeList,  wCoefs, range=plotRange)


SNR.Th <- 3
nearbyPeak <- TRUE
majorPeakInfo <- identifyMajorPeaks(data, ridgeList, wCoefs, SNR.Th = SNR.Th, nearbyPeak=nearbyPeak)

peakIndex <- majorPeakInfo$peakIndex

plotPeak(data, peakIndex, range=plotRange, main=paste('Identified peaks with SNR >', SNR.Th)) 


# another

SNR.Th <- 4
nearbyPeak <- TRUE

peakInfo <- peakDetectionCWT(data, SNR.Th=SNR.Th, nearbyPeak=nearbyPeak)
majorPeakInfo_2 = peakInfo$majorPeakInfo
peakIndex_2 <- majorPeakInfo_2$peakIndex

peakIndex_2 %in% peakIndex

mz_data[peakIndex_2]
mz_data[peakIndex]



plotRange_2 <- c(50, 1600)

plotPeak(data, peakIndex_2, range=plotRange_2, log='x', main=paste('Identified peaks with SNR >', SNR.Th)) 


peakSNR <- majorPeakInfo$peakSNR
allPeakIndex <- majorPeakInfo$allPeakIndex


plotRange <- c(5000, 36000)
selInd <- which(allPeakIndex >= plotRange[1] & allPeakIndex < plotRange[2])
plot(allPeakIndex[selInd], peakSNR[selInd], type='h', xlab='m/z Index', ylab='Signal to Noise Ratio (SNR)', log='x')
points(peakIndex, peakSNR[names(peakIndex)], type='h', col='red')
title('Signal to Noise Ratio (SNR) of the peaks (CWT method)')


betterPeakInfo <- tuneInPeakInfo(data, majorPeakInfo_2)


plotRange <- c(5000, 11000)
plot(plotRange[1]:plotRange[2], data[plotRange[1]:plotRange[2]], type='l', log='x', xlab='m/z Index', ylab='Intensity')
abline(v=betterPeakInfo$peakCenterIndex, col='red')




# try plotting different scans to see the difference
mz76 = rd_list[[76]]@mz
int76 = rd_list[[76]]@intensity

mz1 = rd_list[[1]]@mz
int1 = rd_list[[1]]@intensity

mz2 = rd_list[[2]]@mz
int2 = rd_list[[2]]@intensity

mz8 = rd_list[[8]]@mz
int8 = rd_list[[8]]@intensity

mz36 = rd_list[[36]]@mz
int36 = rd_list[[36]]@intensity

# plot shows tiny differences
plot(mz1,int1,type="l",col="red")
lines(mz76,int76,col="blue")
lines(mz36,int36,col="green")

library(scales)
plot(mz1,int1,type="l",col=alpha("red",0.4))
lines(mz2,int2,col=alpha("blue", 0.4))
lines(mz8,int8,col=alpha("green", 0.2))













