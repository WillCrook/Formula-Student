% TBRe GPS data extension activity:
% Convert coordinates to meters (projecection) to be able to calculate sRun, 
% vCar, gLat and gLong. Find sLap from lap splitting sRun and overlay data 
% from each lap on a lap distance domain.

% Further extension: Explain and show where the fastest lap gains/loses 
% time against the second fastest lap.

% Data from a full FSUK 2025 endurance stint at Silverstone.


%% 1. Load the data
load("Onboarding_GPS.mat")
whos GPS*

%% 2. What does the data look like?
% Time, latitude, longitude, altitude. Lat/lon are stored to 15 decimal
% places, so use format long to see them all.
format long g
firstSamples = [GPStime(1:5); GPSlat(1:5); GPSlon(1:5); GPSalt(1:5)]';
format short

%% 3. Where in the world are we?
% Paste the first position into Google Maps (satellite view).
mapsLink = "https://www.google.com/maps?q=" + num2str(GPSlat(1), 10) + "," + num2str(GPSlon(1), 10) + "&t=k";
web(mapsLink) 

%% 4. Plot the coordinate data
figure
plot(GPSlon, GPSlat)
grid on
xlabel("Longitude (deg)")
ylabel("Latitude (deg)")
title("GPS trace")

%% 5. Define a start/finish line
% A lap begins by crossing a line and ends when the same line is crossed
% again. Two [lon lat] points (chosen by inspection from previous plot 
% define the line.
lineStart = [-1.0111, 52.07812];    % [lon lat]
lineEnd   = [-1.01113, 52.07800];   % [lon lat]
figure
plot(GPSlon, GPSlat)
hold on
plot([lineStart(1) lineEnd(1)], [lineStart(2) lineEnd(2)], "r", LineWidth=3)
hold off
grid on
xlim([-1.01116 -1.01107])
ylim([52.07798 52.07814])
xlabel("Longitude (deg)")
ylabel("Latitude (deg)")
title("Start/finish line (zoomed in)")

%% 6. Crossing detection
% The cross product of the line direction with the car's position is
% negative before the S/F and positive after. The car crosses the line 
% wherever it changes sign between samples.

lineDir = lineEnd - lineStart;
side = lineDir(1) * (GPSlat - lineStart(2)) - lineDir(2) * (GPSlon - lineStart(1));
k = find(side(1:end-1) .* side(2:end) < 0);
numSignChanges = numel(k);

%% 7. Keep only the start/finish crossings
% The cross product line is infinitely long, so the far side of the track
% crosses it heading the other way. Keep crossings only at the S/F

k = k(side(k) < 0);   % negative before the line, positive after
numCrossings = numel(k);

%% 8. Laptime calculation
% Each lap runs from the first sample after one crossing to the first
% sample after the next. The out-lap and in-lap are not timed.
crossTime = GPStime(k + 1);
lapTime = diff(crossTime)';
lap = (1:numel(lapTime))';
lapTimes = table(lap, lapTime);
[fastest, fastestLap] = min(lapTime);
disp("Fastest lap " + fastestLap + " in " + fastest + " s")

%% 9. Line crossing interpolation & more accurate laptimes
% Using the cross product again, interpolate between the samples either 
% side of the line to find the time the car was exactly on it.

frac = side(k) ./ (side(k) - side(k + 1));            % 0 = at sample k, 1 = at sample k+1
crossTimeInterp = GPStime(k) + frac .* (GPStime(k + 1) - GPStime(k));
lapTimeInterp = diff(crossTimeInterp)';
difference = round(lapTimeInterp - lapTime, 3);
comparison = table(lap, lapTime, lapTimeInterp, difference);
[fastestInterp, fastestLapInterp] = min(lapTimeInterp);
disp("Fastest lap interpolated: lap " + fastestLapInterp + " in " + round(fastestInterp, 3) + " s")

%% 10. Extension data: throttle, brake and power
% Throttle is in %, front brake is the brake pressure sensor reading, 
% and power is in kW.

load("Onboarding_rThrottle.mat")   
load("Onboarding_pBrakeF.mat")      
load("Onboarding_Power.mat")  
