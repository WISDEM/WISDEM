function Pl_FastPlots(varargin)
% This function plots outputs from openfast simulations. There is an
% attempt to organize the created plots into some sort of
% categories. 
%
% Inputs: varargin - Some number of structures, should be created from
%                    Post_LoadFastOut.m, or output from a simulink run.
%                    Each structure will be plotted on top of the previous
%                    one.
%
% Nikhar Abbas - February 2019




%% Cases to plot
% Switches to turn on/off some categories of plots. Cases are defined in
% the next section
plsw.MI = 0;                    % MI, Main Inputs
plsw.DTO = 0;                   % DTO, Drivetrain Outputs     
plsw.B1 = 1;                    % B1, Baseline1
plsw.PD = 1;                    % PD, Primary Dynamics
plsw.RO = 0;                    % RO, Rotor Performance Outputs
plsw.Fl1 = 0;                   % Fl1, Basic Floating Parameters
plsw.AF = 0;                    % All Floating Parameters
plsw.Twr = 0;                   % Twr, Turbine params with Twr Motions
plsw.Rand = 0;                  % Some random metrics I care about now
cases = fieldnames(plsw);

%% Plot Cases
% Everything defined here should have a switch above
pc.MI = {'Wind1VelX', 'BldPitch1', 'GenTq'};
pc.DTO = {'GenPwr', 'RotSpeed', 'GenSpeed'};
pc.B1 = {'Wind1VelX', 'BldPitch1', 'GenTq', 'RotSpeed', 'GenPwr'};
pc.PD = {'BldPitch1', 'GenTq', 'GenSpeed'};
pc.RO = {'RtTSR','RtAeroCp'};
pc.Fl1 = {'PtfmPitch', 'BldPitch1'};
pc.AF = {'PtfmPitch', 'PtfmRoll', 'PtfmSurge', 'PtfmYaw', 'PtfmHeave', 'PtfmSway'};
pc.Twr = {'GenTq','BldPitch1','RotSpeed', 'TwrBsFxt'};
pc.Rand = {'RtAeroCt', 'TwrBsFxt', 'LSShftFxs','GenPwr'};

%% load outdata to be plotted
for args = 1:length(varargin)
    outdata(args) = varargin(args);              % load data
end


% Plot!
for dind = 1:length(outdata)
    fo = outdata{dind};
    
    time = fo.Time;
    
    fignum = 100;
    for cind = 1:length(cases)
        if plsw.(cases{cind})
            pcats = pc.(cases{cind});           % Categories to plot              
            subsize = length(pcats);
            
            for plind = 1:length(pcats) 
                fig = figure(fignum); hold on         % Create figure
                subplot(subsize,1,plind)
                try
                    % plot data
                    pdata = fo.(pcats{plind}); % data to plot
                    pl = plot(time,pdata);
                    ylabel(pcats{plind})
                    
%                     if strcmp(pcats{plind},'BldPitch1')
%                         ylim([-5, 45]);         
%                     elseif strcmp(pcats{plind},'RtAeroCp')
%                         ylim([0 0.6]);
%                     end
                    
                    grid on
                    pl.LineWidth = 1.5;
                    
                    if plind == subsize
                        xlabel('Time')
                    end
                catch
                    disp([pcats{plind} ' was not available in the OutList'])            
                end

            end
            fignum = fignum+1;
        end        
    end
end




end