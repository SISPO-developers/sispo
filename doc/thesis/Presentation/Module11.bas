Attribute VB_Name = "Module11"
' Author : Mirza Elahi
' Date : 07 Jul, 2016
Sub AddProgressBar()
    ' Parameters to set
    progressBarHeight = 4.5 ' height of the progress bar
    FillColor = RGB(251, 0, 6) ' Fill color of the progress bar
    LineColor = FillColor ' Line color of the progress bar
    BackgroundColor = RGB(255, 255, 255) ' background color of the progress bar
    fontColor = FillColor
    startingSlideNo = 2
    noFontSize = 13
    showSlideNo = False ' Set this to False if you dont want to show total slide no
    'Slider Making
    On Error Resume Next
        With ActivePresentation
            sHeight = .PageSetup.SlideHeight - progressBarHeight
            n = 0
            j = 0
            For i = 1 To .Slides.Count
                If .Slides(i).SlideShowTransition.Hidden Then j = j + 1
            Next i:
            For i = startingSlideNo To .Slides.Count
                .Slides(i).Shapes("progressBar").Delete
                .Slides(i).Shapes("progressBarBackground").Delete
                .Slides(i).Shapes("pageNumber").Delete
                If .Slides(i).SlideShowTransition.Hidden = msoFalse Then
                    ' Background setting
                    ' Underscore used for continuation of line
                    Set sliderBack = .Slides(i).Shapes.AddShape( _
                                        msoShapeRectangle, 0, _
                                        sHeight, (.Slides.Count - j) _
                                        * .PageSetup.SlideWidth _
                                        / (.Slides.Count - j), _
                                        progressBarHeight)
                    With sliderBack
                        .Fill.ForeColor.RGB = BackgroundColor
                        .Line.ForeColor.RGB = BackgroundColor
                        .Name = "progressBarBackground"
                        End With
                    ' Main Slider setting
                    Set slider = .Slides(i).Shapes.AddShape( _
                                        msoShapeRectangle, 0, _
                                        sHeight, (i - n) * _
                                        .PageSetup.SlideWidth _
                                        / (.Slides.Count - j), _
                                        progressBarHeight)
                    With slider
                        ' enable this line to set theme color
                        '.Fill.ForeColor.RGB = ActivePresentation.SlideMaster.ColorScheme.Colors( _
                        ppFill).RGB
                        .Fill.ForeColor.RGB = FillColor
                        .Line.ForeColor.RGB = LineColor
                        .Name = "progressBar"
                    End With
                    Set pageNumber = .Slides(i).Shapes.AddTextbox( _
                                        msoTextOrientationHorizontal, _
                                        ((.Slides.Count - j) * _
                                        .PageSetup.SlideWidth / _
                                        (.Slides.Count - j)) - 50, _
                                        .PageSetup.SlideHeight - 23, 100, 10)
                    ' Slide No
                    If showSlideNo = True Then
                        With pageNumber
                            .TextFrame.TextRange.Text = Str(i - n) & "/" & _
                                    Str(ActivePresentation.Slides.Count - j)
                            With .TextFrame.TextRange.Font
                                .Bold = msoFalse
                                .Size = noFontSize
                                .Color = fontColor
                            End With
                            .Name = "pageNumber"
                        End With
                    End If

                Else
                    n = n + 1
                End If
            Next i:
        End With
End Sub

Sub RemoveProgressBar()
    On Error Resume Next
    With ActivePresentation
        For i = 1 To .Slides.Count
            .Slides(i).Shapes("progressBar").Delete
            .Slides(i).Shapes("progressBarBackground").Delete
            .Slides(i).Shapes("pageNumber").Delete
        Next i:
    End With
End Sub

