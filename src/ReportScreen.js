import React from "react";
import ReactGoogleSlides from "react-google-slides";
import Typography from '@mui/material/Typography';
import Container from '@mui/material/Container';

function ReportScreen() {
    return (
        <>
            <Container maxWidth="md" sx={{ marginBottom: 10 }}>
                <Typography variant="h5" mb={5}>Project Presentation</Typography>
                <ReactGoogleSlides
                    width={"100%"}
                    slidesLink="https://docs.google.com/presentation/d/1JccFYkKy50q92hSpNuc2Fm_GMKEk0hXG8GEJibCGaIM/edit#slide=id.g25f6af9dd6_0_0"
                    slideDuration={5}
                    position={1}
                    showControls
                    loop
                />
            </Container>
            <Container maxWidth="md" sx={{ marginBottom: 10 }}>
                <Typography variant="h5" mb={5}>Project Report</Typography>
                <object data="http://africau.edu/images/default/sample.pdf" type="application/pdf" width="100%" height="800">
                    <p>Alternative text - include a link <a href="http://africau.edu/images/default/sample.pdf">to the PDF!</a></p>
                </object>
            </Container>
        </>
    )
}

export default ReportScreen;
