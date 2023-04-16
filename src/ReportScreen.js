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
                    slidesLink="https://docs.google.com/presentation/d/1JccFYkKy50q92hSpNuc2Fm_GMKEk0hXG8GEJibCGaIM/edit?usp=sharing"
                    slideDuration={5}
                    position={1}
                    showControls
                    loop
                />
            </Container>
            <Container maxWidth="md" sx={{ marginBottom: 10 }}>
                <Typography variant="h5" mb={5}>Project Report</Typography>
                <iframe src="https://docs.google.com/document/d/1ZqvR6anv8_8e3456rBQH-1fLUr8rbUYEwAXHsstYj30/edit" width="100%" height="800"></iframe>
            </Container>
        </>
    )
}

export default ReportScreen;
