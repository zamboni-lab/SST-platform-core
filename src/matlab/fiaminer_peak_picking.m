
function peaks = fiaminer_peak_picking(args)
    
    path = args.path;
    
%     path = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML';

    mzObj = mzxmlread_2(path);

    out = mzObj.scan(43);

    mz_ratio = out.peaks.mz(1:2:end);
    Y = out.peaks.mz(2:2:end);

    fclose('all');
        
    % mz1 = mzObj.scan(1).peaks.mz(1:2:end);

    % Yb = msbackadj(mzt,Y(I),'WindowSize',0.2,'StepSize',0.2,'PreserveHeights',1,'ShowPlot', 0);

    % [Pt,Wt] = mspeaks(mz_ratio,Y,'SHOWPLOT',0,'denoising',1,'Levels',4,'Base', 2,'HeightFilter',100,'NoiseEstimator','mad','OverSegmentationFilter',mzt(6)-mzt(1),'FWHHFilter',mzt(2)-mzt(1));

    peaks = mspeaks(mz_ratio,Y,'SHOWPLOT',0,'denoising',0,'Levels',1,'Base',2,'HeightFilter',100);
    
%     peaks = [Pt, Wt];
    
end
        
      