import { useState, useEffect } from 'react';
import axios from 'axios';
import { CODE_ATTESTATION, NOTARY_API } from '../utils/constants';
import { RemoteAttestation, generateNonce } from '@freysa/esper-js';
import { OffscreenActionTypes } from '../entries/Offscreen/types';
import { getNotaryConfig } from '../utils/misc';

export const useRemoteAttestation = () => {
  const [remoteAttestation, setRemoteAttestation] =
    useState<RemoteAttestation | null>(null);
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expectedPcrs, setExpectedPcrs] = useState<any>(null);

  useEffect(() => {
    (async () => {
      const config = await getNotaryConfig();
      console.log('expected pcrs', config.EXPECTED_PCRS);
      setExpectedPcrs(config.EXPECTED_PCRS);
    })();
  }, []);

  useEffect(() => {
    (() => {
      chrome.runtime.onMessage.addListener(
        async (request, sender, sendResponse) => {
          switch (request.type) {
            case OffscreenActionTypes.remote_attestation_verification_response: {
              const result = request.data;
              setIsValid(result);
            }
          }
        },
      );
    })();
  }, []);
  useEffect(() => {
    const fetchData = async () => {
      try {
        if (!expectedPcrs) {
          return;
        }
        //const nonce = generateNonce();
        const nonce = '549bef7ffe5e5e3dd6dd050572b036d8e7e092b7';
        // const enclaveEndpoint = `${NOTARY_API.replace(
        //   ':7047',
        //   '',
        // )}/enclave/attestation?nonce=${nonce}`;

        const remoteAttbase64 = CODE_ATTESTATION.trim();

        chrome.runtime.sendMessage({
          type: OffscreenActionTypes.remote_attestation_verification,
          data: {
            remoteAttestation: remoteAttbase64,
            nonce,
            pcrs: expectedPcrs,
          },
        });
      } catch (error) {
        console.log('error fetching code attestation from enclave', error);
        setError(error as any);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [expectedPcrs]);

  return { remoteAttestation, loading, error, isValid };
};
