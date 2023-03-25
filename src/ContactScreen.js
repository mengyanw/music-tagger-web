import React from "react";
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';


function ContactScreen() {
    return (
        <>
            <Container maxWidth="md" sx={{ marginBottom: 60 }}>
                <Typography>
                    Developed by Mengyan Wu (mengyanw@umich.edu), Yuxiao Liu, Zihui Liu
                </Typography>
            </Container>
        </>
    )
}

export default ContactScreen;
