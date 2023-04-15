import React from "react";
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';


function ContactScreen() {
    return (
        <>
            <Container maxWidth="md" sx={{ marginBottom: 60 }}>
                <Typography variant="h6" sx={{ mb: 5 }}>
                    Developed by
                </Typography>
                <Typography variant="body1">
                    Yuxiao Liu (lyuxiao@umich.edu) <br />
                    Zihui Liu (zihuiliu@umich.edu)<br /> 
                    Mengyan Wu (mengyanw@umich.edu) <br /> 
                </Typography>
            </Container>
        </>
    )
}

export default ContactScreen;
