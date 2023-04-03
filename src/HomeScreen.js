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

import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
// import Paper from '@mui/material/Paper';


import { useState } from "react";
import {
    // AUDIO_PATH, MODEL_PATH,
    // Demo,
    LoadMp3, GenerateMelSpec, CropAndFlatten, CreateONNXTensor, RunModel, FinalizeResult
} from "./utils"


function HomeScreen() {
    const [audioPath, setAudioPath] = useState('1.mp3');
    const [modelPath, setModelPath] = useState('samplecnn.pt');
    const [loading, setLoading] = useState(false)
    const [isInitial, setIsInitial] = useState(true)
    const [runningResult, setRunningResult] = useState([]);
    const [processDesc, setProcessDesc] = useState([])
    const [uploadedAudio, setUploadedAudio] = useState({})
    const serviceAudioPath = audioPath === uploadedAudio?.path ? audioPath : '../public/audio/' + audioPath
    const playerAudioPath = audioPath === uploadedAudio?.path ? audioPath : './audio/' + audioPath

    const handleRunningRequest = async () => {
        setRunningResult([])
        setProcessDesc([])
        setLoading(true);
        if (isInitial) setIsInitial(false)

        const formData = new FormData();

        formData.append('audioPath', serviceAudioPath);
        formData.append('modelPath', '../public/model/' + modelPath);
        console.log(formData);

        // setProcessDesc((prev) => [...prev, "Loading MP3 file 🎵"])
        // const audioBuffer = await LoadMp3(audioPath)
        // setProcessDesc((prev) => [...prev, "Resampling and converting signal ⌛️"])
        // setProcessDesc((prev) => [...prev, "Generating mel spectrogram ⌛️"])
        // const melSpec = await GenerateMelSpec(audioBuffer)
        // setProcessDesc((prev) => [...prev, "Cropping and flattening data ⌛️"])
        // const processedData = await CropAndFlatten(melSpec)
        // setProcessDesc((prev) => [...prev, "Creating tensor ⌛️"])
        // const inputTensor = await CreateONNXTensor(processedData)
        // setProcessDesc((prev) => [...prev, "Running model ⌛️"])
        // const outputMap = await RunModel(inputTensor, modelPath)
        // setProcessDesc((prev) => [...prev, "Grabbing results ☕️"])
        // const result = await FinalizeResult(outputMap)
        // setProcessDesc((prev) => [...prev, "Finished 🎉🎉🎉"])

        fetch('http://127.0.0.1:5000/predict2/', {
            method: 'POST',
            body: formData
        })
            .then(res => {
                console.log(res)
                return res.json()
            })
            .then(data => {
                console.log(data);
                setRunningResult(data);
                setLoading(false)
            });

        // setRunningResult(result)
        // setLoading(false)
    }

    return (
        <Container maxWidth="md" sx={{ marginBottom: 10 }}>
            <Container sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
                <Container disableGutters sx={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', }}>
                    <FormControl sx={{ m: 1, minWidth: 200 }} size="medium">
                        <InputLabel id="demo-select-small">Model</InputLabel>
                        <Select
                            labelId="demo-select-small"
                            id="demo-select-small"
                            value={modelPath}
                            label="Model"
                            onChange={(event) => setModelPath(event.target.value)}
                        >
                            <MenuItem value={'fcn.pt'}>FCN model</MenuItem>
                            <MenuItem value={'samplecnn.pt'}>CNN model</MenuItem>
                        </Select>
                    </FormControl>
                    <Container disableGutters sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                        <FormControl sx={{ m: 1, minWidth: 200 }} size="medium">
                            <InputLabel id="demo-select-small">Audio</InputLabel>
                            <Select
                                labelId="demo-select-small"
                                id="demo-select-small"
                                value={audioPath}
                                label="Audio"
                                onChange={(event) => setAudioPath(event.target.value)}
                            >
                                <MenuItem value={'1.mp3'}>Example 1.mp3</MenuItem>
                                <MenuItem value={'2.mp3'}>Example 2.mp3</MenuItem>
                                <MenuItem value={'3.mp3'}>Example 3.mp3</MenuItem>
                                <MenuItem value={'shut_down_blackpink.mp3'}>Shut down (BlackPink).mp3</MenuItem>
                                <MenuItem value={'running_up_that_hill.mp3'}>Running up that hill (Kate Bush).mp3</MenuItem>
                                {/* <MenuItem value={'red_ruby_da_sleeze.mp3'}>Red Ruby Da Sleeze (Nicki Minaj).mp3</MenuItem> */}
                                {uploadedAudio ? <MenuItem value={uploadedAudio.path}>{uploadedAudio.name}</MenuItem> : {}}
                            </Select>
                        </FormControl>
                        <Typography sx={{ m: 1 }} >
                            OR
                        </Typography>
                        <Button
                            variant="outlined"
                            component="label"
                            sx={{ minWidth: 200 }}
                        >
                            Upload audio file
                            <input
                                type="file"
                                hidden
                                accept=".mp3"
                                onChange={(e) => {
                                    let uploadPath = URL.createObjectURL(e.target.files[0])
                                    setAudioPath(uploadPath)
                                    setUploadedAudio({ name: e.target.files[0].name, path: uploadPath })
                                }}
                            />
                        </Button>
                    </Container>
                </Container>
                <audio src={playerAudioPath} controls style={{ width: '90%', padding: '1rem', margin: 'auto' }} />
                <Container disableGutters sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Button
                        variant='contained'
                        onClick={handleRunningRequest}
                        sx={{ minWidth: 200, ml: 1 }}
                    >
                        Run
                    </Button>
                </Container>
            </Container>

            <Box sx={{ flexGrow: 1 }}>
                {/* <Grid container spacing={{ xs: 1, md: 2 }} columns={{ xs: 1, sm: 4 }}> */}
                <Grid container spacing={{ xs: 1, md: 2 }} columns={{ xs: 1, sm: 1 }}>
                    {/* <Grid item xs={1} sm={2} key={0}>
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
                    </Grid> */}
                    <Grid item xs={1} sm={2} key={1}>
                        <Card sx={{ minWidth: 275, minHeight: 420 }}>
                            <CardContent>
                                <Typography sx={{ fontSize: 14, mb: 1.5 }} color="text.secondary" gutterBottom>
                                    Result
                                </Typography>
                                {loading ? <Typography>Loading...</Typography> :
                                    (isInitial ? <div></div> :
                                        <TableContainer component={Paper}>
                                            <Table sx={{ minWidth: 650 }} aria-label="simple table">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell align="center">Category</TableCell>
                                                        <TableCell align="center">Tag</TableCell>
                                                        <TableCell align="center">Probability</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {runningResult.map((row) => (
                                                        <TableRow
                                                            key={row.index}
                                                            sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                                                        >
                                                            <TableCell align="center">{row.Category}</TableCell>
                                                            <TableCell align="center">{row.Tag}</TableCell>
                                                            <TableCell align="center">{row.Probability.toFixed(4)}</TableCell>
                                                        </TableRow>
                                                    ))}
                                                </TableBody>
                                            </Table>
                                        </TableContainer>
                                    )
                                }
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>

            </Box>
        </Container>
    );
}

export default HomeScreen;