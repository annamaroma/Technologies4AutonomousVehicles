function drowsinessState = updateDrowsinessTimers(timestamp, bothEyesClosed, drowsinessState, config)
%UPDATEDROWSINESSTIMERS Update microsleep and sleep timers.

if isnan(drowsinessState.lastTimestamp)
    drowsinessState.lastTimestamp = timestamp;
    return;
end

dt = max(0, timestamp - drowsinessState.lastTimestamp);
drowsinessState.lastTimestamp = timestamp;

if bothEyesClosed
    drowsinessState.closedDuration = drowsinessState.closedDuration + dt;
    drowsinessState.openDuration = 0;
else
    drowsinessState.openDuration = drowsinessState.openDuration + dt;
    drowsinessState.closedDuration = 0;
end

if drowsinessState.closedDuration >= config.microsleepSec
    drowsinessState.microsleepActive = true;
end

if drowsinessState.closedDuration >= config.sleepSec
    drowsinessState.sleepActive = true;
end

if drowsinessState.openDuration >= config.eyesOpenResetSec
    drowsinessState.microsleepActive = false;
    drowsinessState.sleepActive = false;
end

