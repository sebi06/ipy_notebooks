set ORIGINAL_DIR=%CD%
set DIR2PROCESS=c:\Users\m1srh\OneDrive - Carl Zeiss AG\Projects\ATOMIC\CNNs\pypi_czmodel\DemoNotebooks

@REM Set working directory
@ECHO Original Directory: %ORIGINAL_DIR%
@ECHO Working Directory: %DIR2PROCESS%
pushd %DIR2PROCESS%

@REM convert the notebook
call jupyter nbconvert train_simple_TF2_segmentation_model.ipynb --to slides --post serve --SlidesExporter.reveal_theme=serif --SlidesExporter.reveal_scroll=True --SlidesExporter.reveal_transition=slide