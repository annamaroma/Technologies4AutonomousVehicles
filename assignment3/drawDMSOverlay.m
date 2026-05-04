function annotatedFrame = drawDMSOverlay(frame, stateLabel, bpm, diagnostics, config)
%DRAWDMSOVERLAY Draw driver state, BPM and detection diagnostics on frame.

annotatedFrame = frame;

if isfield(diagnostics, "faceBox") && ~isempty(diagnostics.faceBox)
    annotatedFrame = insertObjectAnnotation(annotatedFrame, "rectangle", diagnostics.faceBox, ...
        "Face", "AnnotationColor", "yellow", "FontSize", 12, "LineWidth", 2);
end

if isfield(diagnostics, "noseBox") && ~isempty(diagnostics.noseBox)
    annotatedFrame = insertObjectAnnotation(annotatedFrame, "rectangle", diagnostics.noseBox, ...
        "Nose", "AnnotationColor", "cyan", "FontSize", 10, "LineWidth", 2);
end

if isfield(diagnostics, "leftEyeBox") && ~isempty(diagnostics.leftEyeBox)
    annotatedFrame = insertObjectAnnotation(annotatedFrame, "rectangle", diagnostics.leftEyeBox, ...
        "L eye", "AnnotationColor", "green", "FontSize", 10, "LineWidth", 2);
end

if isfield(diagnostics, "rightEyeBox") && ~isempty(diagnostics.rightEyeBox)
    annotatedFrame = insertObjectAnnotation(annotatedFrame, "rectangle", diagnostics.rightEyeBox, ...
        "R eye", "AnnotationColor", "green", "FontSize", 10, "LineWidth", 2);
end

if isfinite(bpm)
    bpmText = sprintf("HR: %.0f BPM", bpm);
else
    bpmText = "HR: estimating...";
end

imageHeight = size(frame, 1);
imageWidth = size(frame, 2);

annotatedFrame = insertText(annotatedFrame, [10, imageHeight - 42], bpmText, ...
    "FontSize", config.annotationFontSize, ...
    "TextColor", "white", ...
    "BoxColor", "black", ...
    "BoxOpacity", 0.65);

stateTextSize = max(220, 10 * strlength(stateLabel));
annotatedFrame = insertText(annotatedFrame, [imageWidth - stateTextSize, imageHeight - 42], stateLabel, ...
    "FontSize", config.annotationFontSize, ...
    "TextColor", "white", ...
    "BoxColor", localStateColor(stateLabel), ...
    "BoxOpacity", 0.70);

if isfield(diagnostics, "message") && strlength(string(diagnostics.message)) > 0
    annotatedFrame = insertText(annotatedFrame, [10, 10], string(diagnostics.message), ...
        "FontSize", config.warningFontSize, ...
        "TextColor", "white", ...
        "BoxColor", localMessageColor(string(diagnostics.message)), ...
        "BoxOpacity", 0.55);
end

function colorName = localStateColor(stateLabel)
    switch string(stateLabel)
        case "Sleep"
            colorName = "red";
        case "Microsleep"
            colorName = "orange";
        case {"Distracted (long)", "Distracted (short)"}
            colorName = "yellow";
        otherwise
            colorName = "green";
    end
end

function colorName = localMessageColor(messageText)
    if contains(lower(messageText), "calibrating")
        colorName = "blue";
    elseif contains(lower(messageText), "move closer") || contains(lower(messageText), "keep") || ...
            contains(lower(messageText), "hold still") || contains(lower(messageText), "face frontal")
        colorName = "yellow";
    else
        colorName = "red";
    end
end

end
