import React from "react";
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
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
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { useState } from "react";

function HomeScreen() {
    const [audioPath, setAudioPath] = useState('Hey_Jude_The_Beatles.mp3');
    const [modelPath, setModelPath] = useState('musicnn.pt');
    const [loading, setLoading] = useState(false)
    const [isInitial, setIsInitial] = useState(true)
    const [runningResult, setRunningResult] = useState([]);
    const [uploadedAudio, setUploadedAudio] = useState({})
    const serviceAudioPath = audioPath === uploadedAudio?.path ? audioPath : '../public/audio/' + audioPath
    const playerAudioPath = audioPath === uploadedAudio?.path ? audioPath : './audio/' + audioPath

    const handleRunningRequest = async () => {
        setRunningResult([])
        setLoading(true);
        if (isInitial) setIsInitial(false)

        const formData = new FormData();

        formData.append('audioPath', serviceAudioPath);
        formData.append('modelPath', '../public/model/' + modelPath);

        const uploadedAudioFile = audioPath === uploadedAudio?.path ? uploadedAudio?.file : undefined 
        formData.append('uploadedAudio', uploadedAudioFile)
           
        fetch('/api/predict/', {
            method: 'POST',
            body: formData
        })
            .then(res => {
                return res.json()
            })
            .then(data => {
                setRunningResult(data);
                setLoading(false)
            });
    }

    return (
        <Container maxWidth="md" sx={{ marginBottom: 10 }}>
            <Container sx={{ display: 'flex', flexDirection: 'column', mb: 2 }}>
                <Box sx={{ flexGrow: 1 }}>
                    <Grid container columns={{ xs: 1, sm: 4 }}>
                        <Grid item xs={1} sm={2} key={1} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <FormControl sx={{ m: 1, minWidth: 250 }} size="medium">
                                <InputLabel id="demo-select-small">Model</InputLabel>
                                <Select
                                    labelId="demo-select-small"
                                    id="demo-select-small"
                                    value={modelPath}
                                    label="Model"
                                    onChange={(event) => setModelPath(event.target.value)}
                                >
                                    <MenuItem value={'musicnn.pt'}>MusiCNN model <br /> tag precision: 0.8714, auroc: 0.9715</MenuItem>
                                    <MenuItem value={'samplecnn.pt'}>Sample-level CNN model <br />tag precision: 0.5556, auroc: 0.5011</MenuItem>
                                    <MenuItem value={'crnn.pt'}>CRNN model <br />tag precision: 0.3592, auroc: 0.7094</MenuItem>
                                    <MenuItem value={'fcn.pt'}>FCN model <br />tag precision: 0.3156, auroc: 0.6765</MenuItem>
                                    <MenuItem value={'cnnsa.pt'}>CNNSA model <br />tag precision: 0.2905, auroc: 0.6673</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={1} sm={2} key={2}>
                            <Container sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <FormControl sx={{ m: 1, minWidth: 125 }} size="medium">
                                    <InputLabel id="demo-select-small">Audio</InputLabel>
                                    <Select
                                        labelId="demo-select-small"
                                        id="demo-select-small"
                                        value={audioPath}
                                        label="Audio"
                                        onChange={(event) => setAudioPath(event.target.value)}
                                    >
                                        <MenuItem value={'Hey_Jude_The_Beatles.mp3'}>Hey Jude (The Beatles)</MenuItem>
                                        <MenuItem value={'Yellow_Submarine_The_Beatles.mp3'}>Yellow Submarine (The Beatles)</MenuItem>
                                        <MenuItem value={'Hotel_California.mp3'}>Hotel California (Eagles)</MenuItem>
                                        <MenuItem value={'shut_down_blackpink.mp3'}>Shut down (BlackPink)</MenuItem>
                                        <MenuItem value={'running_up_that_hill.mp3'}>Running up that hill (Kate Bush)</MenuItem>
                                        <MenuItem value={'christmas.mp3'}>All I want for Christmas is you (Mariah Carey)</MenuItem>
                                        <MenuItem value={'welcome.mp3'}>Welcome to the jungle (Guns N' Roses)</MenuItem>
                                        <MenuItem value={'asitwas.mp3'}>As it was (Harry Styles)</MenuItem>
                                        <MenuItem value={'wakawaka.mp3'}>Waka Waka (Shakira)</MenuItem>
                                        <MenuItem value={'viva.mp3'}>Viva la vida (Coldplay)</MenuItem>

                                        {uploadedAudio ? <MenuItem value={uploadedAudio.path}>{uploadedAudio.name}</MenuItem> : {}}
                                    </Select>
                                </FormControl>
                                <Button
                                    variant="outlined"
                                    component="label"
                                    sx={{ minWidth: 100 }}
                                >
                                    Upload audio file
                                    <input
                                        type="file"
                                        hidden
                                        accept=".mp3"
                                        onChange={(e) => {
                                            let uploadPath = URL.createObjectURL(e.target.files[0])
                                            setAudioPath(uploadPath)
                                            setUploadedAudio({ name: e.target.files[0].name, path: uploadPath, file: e.target.files[0]  })
                                        }}
                                    />
                                </Button>
                            </Container>
                        </Grid>
                    </Grid>
                </Box>
                <audio src={playerAudioPath} controls style={{ width: '90%', padding: '1rem', margin: 'auto' }} />
                <Container disableGutters sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Button
                        variant='contained'
                        onClick={handleRunningRequest}
                        sx={{ minWidth: 125, ml: 1 }}
                    >
                        Run
                    </Button>
                </Container>
            </Container>

            <Box sx={{ flexGrow: 1 }}>
                <Grid container spacing={{ xs: 1, md: 2 }} columns={{ xs: 1, sm: 1 }}>
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