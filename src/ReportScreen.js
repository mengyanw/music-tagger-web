import React from "react";
import ReactGoogleSlides from "react-google-slides";
import Container from '@mui/material/Container';

function ReportScreen() {
    return (
        <>
            <Container maxWidth="md">
                <ReactGoogleSlides
                    width={"100%"}
                    slidesLink="https://docs.google.com/presentation/d/1E7UU2hlorUFX68dGBQHP8KRNpKB5hXVNodulZvLooSA/edit#slide=id.g25f6af9dd6_0_0"
                    slideDuration={5}
                    position={1}
                    showControls
                    loop
                />
            </Container>
            <Container maxWidth="md">
                <object data="http://africau.edu/images/default/sample.pdf" type="application/pdf" width="100%" height="800">
                    <p>Alternative text - include a link <a href="http://africau.edu/images/default/sample.pdf">to the PDF!</a></p>
                </object>
            </Container>
        </>
    )
}

export default ReportScreen;
