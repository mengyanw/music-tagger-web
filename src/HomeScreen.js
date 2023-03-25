import React from "react";
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';
import Link from '@mui/material/Link';

import { useState } from "react";
import { AUDIO_PATH, MODEL_PATH, Demo } from "./utils";



function HomeScreen() {
    const [runningResult, setRunningResult] = useState([]);
    const handleDemoRequest = async () => {
        const TopNIndex = await Demo(AUDIO_PATH, MODEL_PATH);
        setRunningResult(TopNIndex)
    }
    return (
        <Container maxWidth="md" sx={{ marginBottom: 60 }}>
            <Button variant='outlined' onClick={handleDemoRequest}>Demo</Button>
            {runningResult}
            {/* <Copyright /> */}
        </Container>
        // {/* <ContactScreen /> */}
        // {/* <ReportScreen /> */}
        // <Copyright />
    );
}

export default HomeScreen;