function run_ttide(input_file, output_file, start_time_str, varargin)
    % Load input .mat file
    load(input_file, 'data');

    % Convert string to MATLAB datenum
    time = datenum(start_time_str, 'yyyy-mm-ddTHH:MM:SS');

    % Run t_tide with dynamic optional parameters
    tide = t_tide(data, 'start time', time, varargin{:});

    % Predict tide using same time vector
    npts = length(data);
    dt = 1; % 1 hour assumed
    t = time + (0:dt:(npts-1)*dt) / 24; % hours to days
    predicted = t_predic(t, tide.name, tide.freq, tide.tidecon, varargin{:});

    % Save results
    save(output_file, 'tide', 't', 'predicted');
end
