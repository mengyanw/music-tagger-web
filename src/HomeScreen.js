import React from "react";
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';

import Grid from '@mui/material/Grid';

import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import { styled } from '@mui/material/styles';

import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';

import Player from 'react-material-music-player'
import { Track, PlayerInterface } from 'react-material-music-player'


import { useState } from "react";
import {
    // AUDIO_PATH, MODEL_PATH,
    // Demo,
    LoadMp3, GenerateMelSpec, CropAndFlatten, CreateONNXTensor, RunModel, FinalizeResult
} from "./utils"

const Item = styled(Paper)(({ theme }) => ({
    backgroundColor: theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
    ...theme.typography.body2,
    padding: theme.spacing(1),
    textAlign: 'center',
    color: theme.palette.text.secondary,
}));


function HomeScreen() {
    const [audioPath, setAudioPath] = useState('./1.mp3');
    const [modelPath, setModelPath] = useState('./baseline.onnx');
    const [loading, setLoading] = useState(false)
    const [runningResult, setRunningResult] = useState([]);
    const [processDesc, setProcessDesc] = useState([])

    const currTrack = new Track(audioPath, "./abcover.png", audioPath.slice(2,), "", audioPath)
    PlayerInterface.play([currTrack])

    const handleRunningRequest = async () => {
        setRunningResult([])
        setProcessDesc([])
        setLoading(true);

        setProcessDesc((prev) => [...prev, "Loading MP3 file 🎵"])
        const audioBuffer = await LoadMp3(audioPath)
        setProcessDesc((prev) => [...prev, "Resampling and converting signal ⌛️"])
        setProcessDesc((prev) => [...prev, "Generating mel spectrogram ⌛️"])
        const melSpec = await GenerateMelSpec(audioBuffer)
        setProcessDesc((prev) => [...prev, "Cropping and flattening data ⌛️"])
        const processedData = await CropAndFlatten(melSpec)
        setProcessDesc((prev) => [...prev, "Creating tensor ⌛️"])
        const inputTensor = await CreateONNXTensor(processedData)
        setProcessDesc((prev) => [...prev, "Running model ⌛️"])
        const outputMap = await RunModel(inputTensor, modelPath)
        setProcessDesc((prev) => [...prev, "Grabbing results ☕️"])
        const result = await FinalizeResult(outputMap)
        setProcessDesc((prev) => [...prev, "Finished 🎉🎉🎉"])

        setRunningResult(result)
        setLoading(false)
    }

    return (
        <Container maxWidth="md" sx={{ marginBottom: 10 }}>
            <Container sx={{ display: 'flex', flexDirection: 'row', mb: 2, mr: 5 }}>
                <Container disableGutters>
                    <FormControl sx={{ m: 1, minWidth: 200 }} size="medium">
                        <InputLabel id="demo-select-small">Audio</InputLabel>
                        <Select
                            labelId="demo-select-small"
                            id="demo-select-small"
                            value={audioPath}
                            label="Audio"
                            onChange={(event) => setAudioPath(event.target.value)}
                        >
                            <MenuItem value={'./1.mp3'}>1.mp3</MenuItem>
                            <MenuItem value={'./2.mp3'}>2.mp3</MenuItem>
                            <MenuItem value={'./3.mp3'}>3.mp3</MenuItem>
                            <MenuItem value={'./shut_down_blackpink.mp3'}>Shut down (BlackPink).mp3</MenuItem>
                            <MenuItem value={'./running_up_that_hill.mp3'}>Running up that hill (Kate Bush).mp3</MenuItem>
                            <MenuItem value={'./red_ruby_da_sleeze.mp3'}>Red Ruby Da Sleeze (Nicki Minaj).mp3</MenuItem>

                        </Select>
                    </FormControl>
                    <FormControl sx={{ m: 1, minWidth: 200 }} size="medium">
                        <InputLabel id="demo-select-small">Model</InputLabel>
                        <Select
                            labelId="demo-select-small"
                            id="demo-select-small"
                            value={modelPath}
                            label="Model"
                            onChange={(event) => setModelPath(event.target.value)}
                        >
                            <MenuItem value={'./baseline.onnx'}>Baseline model</MenuItem>
                            {/* <MenuItem value={'./crnn.onnx'}>CRNN model</MenuItem>  */}
                            <MenuItem value={'./test.onnx'}>Test model</MenuItem>
                        </Select>
                    </FormControl>
                </Container>

                <Button
                    variant='outlined'
                    onClick={handleRunningRequest}
                    sx={{ width: '10rem', mr: 1 }}
                >
                    Run
                </Button>
            </Container>

            <Player sx={{ width: '53em', display: 'block', position: 'relative', mb: 5, boxShadow: 1 }} disableDrawer={false} />

            <Box sx={{ flexGrow: 1 }}>
                <Grid container spacing={{ xs: 1, md: 2 }} columns={{ xs: 1, sm: 4 }}>
                    <Grid item xs={1} sm={2} key={0}>
                        <Card sx={{ minWidth: 275, minHeight: 420 }}>
                            <CardContent>
                                <Typography sx={{ fontSize: 14, mb: 1.5 }} color="text.secondary" gutterBottom>
                                    Progress
                                </Typography>
                                <List dense={true}>
                                    {processDesc && processDesc.map((line, idx) => (
                                        <ListItem key={idx + 1}>
                                            <ListItemText
                                                primary={line}
                                            />
                                        </ListItem>
                                    ))}
                                    <ListItem key={0}>
                                        {loading && <CircularProgress size="1.5rem" />}
                                    </ListItem>
                                </List>
                            </CardContent>
                        </Card>
                    </Grid>
                    <Grid item xs={1} sm={2} key={1}>
                        <Card sx={{ minWidth: 275, minHeight: 420 }}>
                            <CardContent>
                                <Typography sx={{ fontSize: 14, mb: 1.5 }} color="text.secondary" gutterBottom>
                                    Result
                                </Typography>
                                <Stack direction="column" spacing={4}>
                                {runningResult !== [] &&
                                    runningResult.map(each => <Item key={each}>{each}</Item>)
                                }
                                </Stack>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            </Box>
        </Container>
    );
}

export default HomeScreen;